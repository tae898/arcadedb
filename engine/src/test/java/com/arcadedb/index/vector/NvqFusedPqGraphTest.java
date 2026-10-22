/*
 * Copyright © 2021-present Arcade Data Ltd (info@arcadedata.com)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: 2021-present Arcade Data Ltd (info@arcadedata.com)
 * SPDX-License-Identifier: Apache-2.0
 */
package com.arcadedb.index.vector;

import com.arcadedb.TestHelper;
import com.arcadedb.database.MutableDocument;
import com.arcadedb.database.RID;
import com.arcadedb.index.TypeIndex;
import com.arcadedb.utility.Pair;
import io.github.jbellis.jvector.graph.disk.OnDiskGraphIndex;
import io.github.jbellis.jvector.graph.disk.feature.FeatureId;
import io.github.jbellis.jvector.graph.disk.feature.NVQ;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

/**
 * The {@code nvq} option of a PRODUCT vector index: the graph file holds jvector's NVQ vectors (8-bit non-uniform quantization) in place of the inline vectors,
 * and a search walks on the PQ codes and re-ranks on the NVQ vectors through jvector's own {@code ScoringView}, without reading a record. With {@code fusedPQ}
 * as well the walk is on the codes fused into the nodes: jvector's intended layout, the one {@code Grid} of jvector-examples benchmarks.
 * <p>
 * 20,000 vectors of 64 dimensions in 100 clusters, a fixed seed: the graph file spans many pages and its nodes are not a divisor of a page, so nodes cross
 * page boundaries (the case of {@link Issue10514FusedPqTest}).
 */
@Tag("vector")
class NvqFusedPqGraphTest extends TestHelper {
  private static final int DIMENSIONS = 64;
  private static final int VECTORS    = 20_000;
  private static final int CLUSTERS   = 100;
  private static final int QUERIES    = 100;
  private static final int K          = 10;
  private static final int EF_SEARCH  = 100;
  private static final int RERANK     = 10;

  @Test
  void aFusedNvqIndexIsSearchedThroughItsFileWithoutReadingARecordAndSurvivesReopenOverlayDeletesAndRebuild() throws Exception {
    final float[][] base = clustered(new Random(10514L), VECTORS);
    final float[][] queries = clustered(new Random(105140L), QUERIES);
    createType("DocFused", "\"fusedPQ\": true");
    createType("DocNvq", "\"fusedPQ\": true, \"nvq\": true");
    insert("DocFused", base);
    insert("DocNvq", base);
    indexOf("DocFused").buildVectorGraphNow();
    indexOf("DocNvq").buildVectorGraphNow();

    // The session that built the index searches through the file it wrote (the graph is reloaded as an OnDiskGraphIndex, as for storeVectorsInGraph)
    assertThat(indexOf("DocNvq").getGraphIndex()).isInstanceOf(OnDiskGraphIndex.class);
    assertThat(indexOf("DocFused").getGraphIndex()).as("no reload without nvq: the resident graph stays").isNotInstanceOf(OnDiskGraphIndex.class);
    final long fusedBytes = indexOf("DocFused").getGraphFile().getLastWrittenGraphBytes();
    final long nvqBytes = indexOf("DocNvq").getGraphFile().getLastWrittenGraphBytes();

    database.close();
    database = factory.open();
    final LSMVectorIndex fused = indexOf("DocFused");
    final LSMVectorIndex nvq = indexOf("DocNvq");
    assertThat(nvq.metadata.nvq).as("the setting survives the reopen").isTrue();
    assertThat(fused.metadata.nvq).isFalse();

    // The features of the two files: NVQ takes the slot of the inline vectors and FusedPQ stays
    final int nvqNodeBytes;
    try (final OnDiskGraphIndex graph = nvq.getGraphFile().loadGraph()) {
      assertThat(graph.getFeatureSet()).containsExactlyInAnyOrder(FeatureId.NVQ_VECTORS, FeatureId.FUSED_PQ);
      assertThat(graph.getIdUpperBound()).isEqualTo(VECTORS);
      assertThat(nvq.getGraphFile().getTotalPages()).as("pages of the graph file").isGreaterThan(30);
      nvqNodeBytes = ((NVQ) graph.getFeatures().get(FeatureId.NVQ_VECTORS)).getNVQuantization().compressedVectorSize();
    }
    try (final OnDiskGraphIndex graph = fused.getGraphFile().loadGraph()) {
      assertThat(graph.getFeatureSet()).as("a file written without nvq is unchanged").containsExactlyInAnyOrder(FeatureId.INLINE_VECTORS, FeatureId.FUSED_PQ);
    }
    // The NVQ copy is extra and smaller than fp32: the file grows by one compressed vector per node
    assertThat(nvqNodeBytes).as("NVQ bytes per node against %d of float32", DIMENSIONS * Float.BYTES).isLessThan((int) (DIMENSIONS * Float.BYTES * 0.6));
    assertThat(nvqBytes - fusedBytes).as("graph file growth (fused %d, fused+nvq %d)", fusedBytes, nvqBytes)
        .isBetween((long) (VECTORS * nvqNodeBytes * 0.98), (long) (VECTORS * nvqNodeBytes * 1.02));

    // The first search of the session: the NVQ file re-ranks without a record read, the file without NVQ reads the records of its candidates
    final long nvqReads = nvq.metrics.getVectorFetchFromDocuments();
    final long fusedReads = fused.metrics.getVectorFetchFromDocuments();
    assertThat(nvq.findNeighborsFromVectorApproximate(queries[0], K, EF_SEARCH, RERANK)).hasSize(K);
    assertThat(fused.findNeighborsFromVectorApproximate(queries[0], K, EF_SEARCH, RERANK)).hasSize(K);
    assertThat(nvq.metrics.getVectorFetchFromDocuments() - nvqReads).as("records read to re-rank on NVQ").isZero();
    assertThat(fused.metrics.getVectorFetchFromDocuments() - fusedReads).as("records read to re-rank on fp32").isGreaterThan(0L);

    double fusedRecall = 0, nvqRecall = 0, worstRelativeError = 0;
    final long scansBefore = nvq.metrics.getBruteForceScans();
    for (final float[] query : queries) {
      final Set<Integer> truth = nearest(base, query, K);
      final List<Pair<RID, Float>> answer = nvq.findNeighborsFromVectorApproximate(query, K, EF_SEARCH, RERANK);
      assertThat(answer).as("a search of the NVQ file returns k distinct rows").hasSize(K);
      assertThat(answer.stream().map(Pair::getFirst).distinct().count()).isEqualTo(K);
      nvqRecall += recall(truth, answer);
      fusedRecall += recall(truth, fused.findNeighborsFromVectorApproximate(query, K, EF_SEARCH, RERANK));
      // The distances of a re-ranked answer are NVQ estimates of the squared L2 distance of the stored vector, ascending
      float previous = -1;
      for (final Pair<RID, Float> row : answer) {
        final double exact = squaredDistance(query, base[vid(row.getFirst())]);
        worstRelativeError = Math.max(worstRelativeError, Math.abs(row.getSecond() - exact) / Math.max(exact, 1.0));
        assertThat(row.getSecond()).isGreaterThanOrEqualTo(previous);
        previous = row.getSecond();
      }
    }
    assertThat(nvq.metrics.getBruteForceScans() - scansBefore).as("brute-force scans that topped up a short walk").isZero();
    nvqRecall /= QUERIES;
    fusedRecall /= QUERIES;
    assertThat(fusedRecall).as("the fp32 re-rank of the fused file").isGreaterThan(0.9);
    assertThat(nvqRecall).as("NVQ re-rank recall (fp32 re-rank %.4f)", fusedRecall).isGreaterThan(fusedRecall - 0.02);
    assertThat(worstRelativeError).as("worst relative error of an NVQ distance").isLessThan(0.05);

    // A search inside a transaction sees its own uncommitted row: scored exactly, it beats the NVQ-scored rows of the graph by the NVQ error only
    final float[] probe = queries[0].clone();
    database.begin();
    final MutableDocument pending = database.newDocument("DocNvq");
    pending.set("vid", VECTORS);
    pending.set("embedding", probe);
    pending.save();
    final List<Pair<RID, Float>> inTransaction = nvq.findNeighborsFromVectorApproximate(probe, K, EF_SEARCH, RERANK);
    assertThat(inTransaction).hasSize(K);
    assertThat(inTransaction.get(0).getFirst()).as("the row the transaction wrote is the nearest to its own vector").isEqualTo(pending.getIdentity());
    assertThat(inTransaction.get(0).getSecond()).isLessThan(1e-3f);
    assertThat(inTransaction.stream().map(Pair::getFirst).distinct().count()).isEqualTo(K);
    database.rollback();
    assertThat(nvq.findNeighborsFromVectorApproximate(probe, K, EF_SEARCH, RERANK).stream().map(Pair::getFirst))
        .as("after the rollback").doesNotContain(pending.getIdentity());

    // Deletes: a row removed in the transaction is gone from its answers at once, and from everybody's after the commit
    final List<RID> doomed = new ArrayList<>();
    final Set<Integer> doomedVids = new HashSet<>();
    for (final Pair<RID, Float> row : nvq.findNeighborsFromVectorApproximate(queries[1], 5, EF_SEARCH, RERANK)) {
      doomed.add(row.getFirst());
      doomedVids.add(vid(row.getFirst()));
    }
    database.begin();
    for (final RID rid : doomed)
      database.lookupByRID(rid, true).delete();
    assertThat(nvq.findNeighborsFromVectorApproximate(queries[1], K, EF_SEARCH, RERANK)).hasSize(K)
        .extracting(Pair::getFirst).doesNotContainAnyElementsOf(doomed);
    database.commit();
    assertThat(nvq.findNeighborsFromVectorApproximate(queries[1], K, EF_SEARCH, RERANK)).hasSize(K)
        .extracting(Pair::getFirst).doesNotContainAnyElementsOf(doomed);

    // REBUILD INDEX writes the file again, over the live rows only, with the same features, and the answers stay as good
    database.command("sql", "REBUILD INDEX `DocNvq[embedding]`").close();
    indexOf("DocNvq").buildVectorGraphNow();
    database.close();
    database = factory.open();
    final LSMVectorIndex rebuilt = indexOf("DocNvq");
    try (final OnDiskGraphIndex graph = rebuilt.getGraphFile().loadGraph()) {
      assertThat(graph.getFeatureSet()).containsExactlyInAnyOrder(FeatureId.NVQ_VECTORS, FeatureId.FUSED_PQ);
      assertThat(graph.getIdUpperBound()).isBetween(VECTORS - 100, VECTORS);
    }
    double rebuiltRecall = 0;
    for (final float[] query : queries) {
      final List<Pair<RID, Float>> answer = rebuilt.findNeighborsFromVectorApproximate(query, K, EF_SEARCH, RERANK);
      assertThat(answer).hasSize(K);
      assertThat(answer.stream().map(Pair::getFirst)).doesNotContainAnyElementsOf(doomed);
      // The truth over the rows that are left
      rebuiltRecall += recall(nearest(base, query, K, doomedVids), answer);
    }
    assertThat(rebuiltRecall / QUERIES).as("recall after REBUILD INDEX").isGreaterThan(0.9);
  }

  @Test
  void nvqWithoutFusedPqWalksOnTheResidentCodesAndReRanksOnNvq() throws Exception {
    final float[][] base = clustered(new Random(7L), 4000);
    final float[][] queries = clustered(new Random(8L), 30);
    createType("DocNvq", "\"nvq\": true");
    insert("DocNvq", base);
    indexOf("DocNvq").buildVectorGraphNow();
    database.close();
    database = factory.open();
    final LSMVectorIndex index = indexOf("DocNvq");
    try (final OnDiskGraphIndex graph = index.getGraphFile().loadGraph()) {
      assertThat(graph.getFeatureSet()).containsExactly(FeatureId.NVQ_VECTORS);
    }
    final long reads = index.metrics.getVectorFetchFromDocuments();
    double recall = 0;
    for (final float[] query : queries) {
      final List<Pair<RID, Float>> answer = index.findNeighborsFromVectorApproximate(query, K, EF_SEARCH, RERANK);
      assertThat(answer).hasSize(K);
      recall += recall(nearest(base, query, K), answer);
    }
    assertThat(index.metrics.getVectorFetchFromDocuments() - reads).as("records read").isZero();
    assertThat(recall / queries.length).isGreaterThan(0.9);
    // rerank 0 asks for the PQ scores alone: no NVQ, no record
    assertThat(index.findNeighborsFromVectorApproximate(queries[0], K, EF_SEARCH, 0)).hasSize(K);
    assertThat(index.metrics.getVectorFetchFromDocuments() - reads).isZero();
  }

  @Test
  void aPqCodebookThatDoesNotFitStillGivesAnNvqFileThatTheExactWalkSearches() throws Exception {
    // Fewer vectors than PQ clusters: no codebook, so the search is the exact walk, on a file whose only vector feature is NVQ
    final float[][] base = clustered(new Random(3L), 100);
    createType("DocNvq", "\"nvq\": true");
    insert("DocNvq", base);
    indexOf("DocNvq").buildVectorGraphNow();
    database.close();
    database = factory.open();
    final LSMVectorIndex index = indexOf("DocNvq");
    assertThat(index.isPQSearchAvailable()).isFalse();
    try (final OnDiskGraphIndex graph = index.getGraphFile().loadGraph()) {
      assertThat(graph.getFeatureSet()).containsExactly(FeatureId.NVQ_VECTORS);
    }
    final float[] query = clustered(new Random(4L), 1)[0];
    assertThat(index.findNeighborsFromVector(query, K, EF_SEARCH, null)).hasSize(K);
    assertThat(recall(nearest(base, query, K), index.findNeighborsFromVector(query, K, EF_SEARCH, null))).isGreaterThan(0.8);
  }

  @Test
  void cosineDistancesOfAnNvqAnswerAreOnTheCosineScale() throws Exception {
    final int dimensions = 32;
    final Random random = new Random(11L);
    final float[][] base = new float[3000][dimensions];
    for (final float[] vector : base)
      for (int j = 0; j < dimensions; j++)
        vector[j] = (float) random.nextGaussian();
    database.transaction(() -> {
      database.command("sql", "CREATE DOCUMENT TYPE DocCos");
      database.command("sql", "CREATE PROPERTY DocCos.vid INTEGER");
      database.command("sql", "CREATE PROPERTY DocCos.embedding ARRAY_OF_FLOATS");
      database.command("sql", "CREATE INDEX ON DocCos (embedding) LSM_VECTOR METADATA { \"dimensions\": " + dimensions
          + ", \"similarity\": \"COSINE\", \"quantization\": \"PRODUCT\", \"nvq\": true, \"addHierarchy\": false }");
    });
    insert("DocCos", base);
    indexOf("DocCos").buildVectorGraphNow();
    database.close();
    database = factory.open();
    final LSMVectorIndex index = indexOf("DocCos");
    for (int q = 0; q < 10; q++) {
      final float[] query = base[q * 7];
      final List<Pair<RID, Float>> answer = index.findNeighborsFromVectorApproximate(query, K, EF_SEARCH, RERANK);
      assertThat(answer).hasSize(K);
      for (final Pair<RID, Float> row : answer) {
        final float[] vector = base[vid(row.getFirst())];
        double dot = 0, a = 0, b = 0;
        for (int j = 0; j < dimensions; j++) {
          dot += query[j] * vector[j];
          a += query[j] * query[j];
          b += vector[j] * vector[j];
        }
        // COSINE distance of this engine is 1 - cosine
        assertThat((double) row.getSecond()).isCloseTo(1.0 - dot / Math.sqrt(a * b), org.assertj.core.data.Offset.offset(0.03));
      }
    }
  }

  @Test
  void nvqIsRefusedWithStoreVectorsInGraphAndIgnoredWithoutProductQuantization() throws Exception {
    assertThatThrownBy(() -> database.transaction(() -> {
      database.command("sql", "CREATE DOCUMENT TYPE DocBoth");
      database.command("sql", "CREATE PROPERTY DocBoth.embedding ARRAY_OF_FLOATS");
      database.command("sql", "CREATE INDEX ON DocBoth (embedding) LSM_VECTOR METADATA { \"dimensions\": " + DIMENSIONS
          + ", \"similarity\": \"EUCLIDEAN\", \"quantization\": \"PRODUCT\", \"nvq\": true, \"storeVectorsInGraph\": true }");
    })).hasStackTraceContaining("nvq and storeVectorsInGraph");

    // Not PRODUCT: the key is accepted and the file is what it always was
    database.transaction(() -> {
      database.command("sql", "CREATE DOCUMENT TYPE DocNone");
      database.command("sql", "CREATE PROPERTY DocNone.vid INTEGER");
      database.command("sql", "CREATE PROPERTY DocNone.embedding ARRAY_OF_FLOATS");
      database.command("sql", "CREATE INDEX ON DocNone (embedding) LSM_VECTOR METADATA { \"dimensions\": " + DIMENSIONS
          + ", \"similarity\": \"EUCLIDEAN\", \"nvq\": true, \"addHierarchy\": false }");
    });
    final float[][] base = clustered(new Random(5L), 600);
    insert("DocNone", base);
    indexOf("DocNone").buildVectorGraphNow();
    try (final OnDiskGraphIndex graph = indexOf("DocNone").getGraphFile().loadGraph()) {
      assertThat(graph.getFeatureSet()).containsExactly(FeatureId.INLINE_VECTORS);
    }
    assertThat(indexOf("DocNone").findNeighborsFromVector(base[0], K, EF_SEARCH, null)).hasSize(K);
  }

  private void createType(final String type, final String extraMetadata) {
    database.transaction(() -> {
      database.command("sql", "CREATE DOCUMENT TYPE " + type);
      database.command("sql", "CREATE PROPERTY " + type + ".vid INTEGER");
      database.command("sql", "CREATE PROPERTY " + type + ".embedding ARRAY_OF_FLOATS");
      database.command("sql", "CREATE INDEX ON " + type + " (embedding) LSM_VECTOR METADATA { \"dimensions\": " + DIMENSIONS
          + ", \"similarity\": \"EUCLIDEAN\", \"quantization\": \"PRODUCT\", \"storeVectorsInGraph\": false, \"addHierarchy\": false, "
          + extraMetadata + " }");
    });
  }

  private void insert(final String type, final float[][] base) {
    database.transaction(() -> {
      for (int i = 0; i < base.length; i++) {
        final MutableDocument doc = database.newDocument(type);
        doc.set("vid", i);
        doc.set("embedding", base[i]);
        doc.save();
      }
    });
  }

  private LSMVectorIndex indexOf(final String type) {
    return (LSMVectorIndex) ((TypeIndex) database.getSchema().getIndexByName(type + "[embedding]")).getIndexesOnBuckets()[0];
  }

  private int vid(final RID rid) {
    return database.lookupByRID(rid, true).asDocument().getInteger("vid");
  }

  private double recall(final Set<Integer> truth, final List<Pair<RID, Float>> answer) {
    int hits = 0;
    for (final Pair<RID, Float> neighbour : answer)
      if (truth.contains(vid(neighbour.getFirst())))
        hits++;
    return (double) hits / K;
  }

  /** {@code count} vectors around {@link #CLUSTERS} centres; the centres come from their own fixed seed, so base and queries share them. */
  private static float[][] clustered(final Random random, final int count) {
    final Random centreRandom = new Random(42L);
    final float[][] centres = new float[CLUSTERS][DIMENSIONS];
    for (final float[] centre : centres)
      for (int j = 0; j < DIMENSIONS; j++)
        centre[j] = (float) (centreRandom.nextGaussian() * 3.0);
    final float[][] vectors = new float[count][DIMENSIONS];
    for (int i = 0; i < count; i++) {
      final float[] centre = centres[random.nextInt(CLUSTERS)];
      for (int j = 0; j < DIMENSIONS; j++)
        vectors[i][j] = centre[j] + (float) random.nextGaussian();
    }
    return vectors;
  }

  private static Set<Integer> nearest(final float[][] base, final float[] query, final int k) {
    return nearest(base, query, k, Set.of());
  }

  private static Set<Integer> nearest(final float[][] base, final float[] query, final int k, final Set<Integer> excluded) {
    final double[] distance = new double[base.length];
    final List<Integer> order = new ArrayList<>(base.length);
    for (int i = 0; i < base.length; i++) {
      distance[i] = squaredDistance(query, base[i]);
      if (!excluded.contains(i))
        order.add(i);
    }
    order.sort(Comparator.comparingDouble(i -> distance[i]));
    return new HashSet<>(order.subList(0, k));
  }

  private static float squaredDistance(final float[] a, final float[] b) {
    double sum = 0;
    for (int i = 0; i < a.length; i++) {
      final double d = a[i] - b[i];
      sum += d * d;
    }
    return (float) sum;
  }
}
