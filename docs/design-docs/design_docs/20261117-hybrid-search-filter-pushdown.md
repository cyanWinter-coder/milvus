# Hybrid Search Filter Pushdown and Read Consistency

## Status

Implementation and integration coverage are included in this change. The
public Hybrid Search request shape is unchanged.

## Summary

Hybrid Search is a routing and fusion request, not a scalar query of its own.
Each entry in `HybridSearchRequest.requests` carries its own `dsl` and
`expr_template_values`. Proxy compiles every sub-request into an independent
query plan, and QueryNode applies that plan before ANN candidate selection.

This change closes two gaps in that path:

1. Proxy now aggregates the optimizable-PK hint across all sub-requests. A
   selective branch can therefore drive QueryNode segment pruning instead of
   being ignored because another branch has no PK predicate.
2. Advanced Search fan-out now gives every sub-search an independent segment
   snapshot. Segment pruning mutates the snapshot in place; sharing it across
   concurrent branches let one branch hide segments needed by its siblings.

The existing request-level consistency and pagination semantics are preserved.

## API Contract

No public protobuf field is added.

```text
HybridSearchRequest
  requests[i].dsl                    scalar filter for retrieval branch i
  requests[i].expr_template_values   template values for that filter
  requests[i].search_params          topk, metric, ANN parameters
  rank_params.limit / offset         final pagination after fusion
  consistency_level                  one read level for the whole request
  guarantee_timestamp                one logical read snapshot for the request
```

There is no top-level `filter` parameter. Passing a top-level filter to an SDK
is not an alternative to per-sub-request `expr`; use `expr` on every
`AnnSearchRequest`.

## Execution Invariants

- Proxy parses each sub-request independently and serializes its filter into
  that sub-request's `SerializedExprPlan`.
- The parent PK hint is the union of optimizable sub-request predicates. A
  positive hint only enables pruning; QueryNode still validates each plan and
  leaves non-PK branches untouched.
- QueryNode resolves one `guarantee_timestamp`/MVCC snapshot per Hybrid Search
  request before fan-out. All branches read the same snapshot.
- Delete visibility is evaluated by segcore inside that snapshot. A deleted PK
  cannot be reintroduced by fusion, pagination, or a retry of another branch.
- Each branch receives a private copy of the pinned segment list because
  clustering-key and PK pruning are in-place operations.
- `limit`/`offset` are applied once after fusion. Callers should make each
  branch fetch at least the parent `limit + offset` candidates so page
  boundaries stay stable.

## Error Handling

Invalid expressions remain input errors from the existing parser
(`merr.WrapErrParameterInvalid*`). Missing or invalid ANN parameters remain
input errors. QueryNode snapshot/pruning failures are system errors and remain
retryable; they are never converted to empty results.

## Tests

Focused Go tests:

```bash
go test -tags dynamic,test -gcflags="all=-N -l" -count=1 ./internal/proxy/dql/... -run 'TestAdvancedSearchAggregatesPKFilterHint|TestCheckSegmentFilter'
go test -tags dynamic,test -gcflags="all=-N -l" -count=1 ./internal/querynodev2/delegator/... -run TestCloneSnapshotItemsIsolatesSubSearches
go test -tags dynamic,test -gcflags="all=-N -l" -count=1 ./internal/views/viewquery/... -run TestBuildAndUpdateSubSearchRequest
```

Integration test (requires the MiniCluster dependencies described in
`tests/integration/README.md`):

```bash
source scripts/setenv.sh
cd tests/integration/hellomilvus
go test -tags dynamic,test -gcflags="all=-N -l" -count=1 -run 'TestHelloMilvus/TestHybridSearchFilterPushdown' -v
```

The integration case covers:

- scalar + PK-range filtering against a reference `Query`;
- Strong/Bounded/Eventually consistency requests;
- deletion visibility after the delete is acknowledged;
- offset pagination with no duplicate or omitted IDs;
- empty filtered results.

## Performance

Filter pushdown avoids full segment scans when the predicate proves a segment
cannot match. The pinned-snapshot copy is O(materialized segment entries); it
allocates one slice per node group and does not copy segment data.

Measure latency in the target deployment before claiming a percentage
improvement. The required 20% target is workload-dependent: it is most likely
when a selective clustering-key or PK predicate removes a large fraction of
sealed segments. No synthetic benchmark result is asserted in CI.

## Known Limitations

- Hybrid Search has no top-level filter; filters must be supplied per
  sub-request.
- QueryNode applies `limit`/`offset` after fusion. Pagination across separate
  requests is stable only when the caller keeps the same consistency/snapshot
  semantics and the collection is not changing between pages.
- Set each sub-request `topk` large enough to cover the deepest page
  (`parent.offset + parent.limit`). Hybrid does not silently grow a sub-request
  beyond the caller's ANN limit; a too-small branch window can change fusion
  candidates between pages.
- Eventually and Bounded reads may observe older state by definition. Strong
  reads are the reference for delete exclusion.
- A positive PK hint does not prove every sub-request is optimizable. QueryNode
  validates each plan and falls back to scanning that branch when necessary.
