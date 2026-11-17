package dql

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/proto/internalpb"
	"github.com/milvus-io/milvus/pkg/v3/proto/planpb"
	"github.com/milvus-io/milvus/pkg/v3/util/paramtable"
	"github.com/milvus-io/milvus/pkg/v3/util/timerecord"
)

func TestCheckSegmentFilter(t *testing.T) {
	t.Run("pk term expr", func(t *testing.T) {
		plan := &planpb.PlanNode{
			Node: &planpb.PlanNode_Query{
				Query: &planpb.QueryPlanNode{
					Predicates: &planpb.Expr{
						Expr: &planpb.Expr_TermExpr{
							TermExpr: &planpb.TermExpr{
								ColumnInfo: &planpb.ColumnInfo{
									IsPrimaryKey: true,
									DataType:     schemapb.DataType_Int64,
								},
								Values: []*planpb.GenericValue{
									{Val: &planpb.GenericValue_Int64Val{Int64Val: 1}},
								},
							},
						},
					},
				},
			},
		}
		assert.Equal(t, common.PkFilterHasPkFilter, checkSegmentFilter(plan))
	})

	t.Run("pk unary range expr", func(t *testing.T) {
		plan := &planpb.PlanNode{
			Node: &planpb.PlanNode_Query{
				Query: &planpb.QueryPlanNode{
					Predicates: &planpb.Expr{
						Expr: &planpb.Expr_UnaryRangeExpr{
							UnaryRangeExpr: &planpb.UnaryRangeExpr{
								ColumnInfo: &planpb.ColumnInfo{
									IsPrimaryKey: true,
									DataType:     schemapb.DataType_Int64,
								},
								Op:    planpb.OpType_GreaterThan,
								Value: &planpb.GenericValue{Val: &planpb.GenericValue_Int64Val{Int64Val: 100}},
							},
						},
					},
				},
			},
		}
		assert.Equal(t, common.PkFilterHasPkFilter, checkSegmentFilter(plan))
	})

	t.Run("non pk predicate", func(t *testing.T) {
		plan := &planpb.PlanNode{
			Node: &planpb.PlanNode_Query{
				Query: &planpb.QueryPlanNode{
					Predicates: &planpb.Expr{
						Expr: &planpb.Expr_TermExpr{
							TermExpr: &planpb.TermExpr{
								ColumnInfo: &planpb.ColumnInfo{
									IsPrimaryKey: false,
									DataType:     schemapb.DataType_Int64,
								},
								Values: []*planpb.GenericValue{
									{Val: &planpb.GenericValue_Int64Val{Int64Val: 1}},
								},
							},
						},
					},
				},
			},
		}
		assert.Equal(t, common.PkFilterNoPkFilter, checkSegmentFilter(plan))
	})

	t.Run("nil plan node", func(t *testing.T) {
		plan := &planpb.PlanNode{}
		assert.Equal(t, common.PkFilterNoPkFilter, checkSegmentFilter(plan))
	})

	t.Run("nil predicates", func(t *testing.T) {
		plan := &planpb.PlanNode{
			Node: &planpb.PlanNode_Query{
				Query: &planpb.QueryPlanNode{},
			},
		}
		assert.Equal(t, common.PkFilterNoPkFilter, checkSegmentFilter(plan))
	})

	t.Run("VectorAnns plan with pk predicate", func(t *testing.T) {
		plan := &planpb.PlanNode{
			Node: &planpb.PlanNode_VectorAnns{
				VectorAnns: &planpb.VectorANNS{
					Predicates: &planpb.Expr{
						Expr: &planpb.Expr_TermExpr{
							TermExpr: &planpb.TermExpr{
								ColumnInfo: &planpb.ColumnInfo{
									IsPrimaryKey: true,
									DataType:     schemapb.DataType_Int64,
								},
								Values: []*planpb.GenericValue{
									{Val: &planpb.GenericValue_Int64Val{Int64Val: 1}},
								},
							},
						},
					},
				},
			},
		}
		assert.Equal(t, common.PkFilterHasPkFilter, checkSegmentFilter(plan))
	})

	t.Run("nested AND with pk on one side", func(t *testing.T) {
		plan := &planpb.PlanNode{
			Node: &planpb.PlanNode_Query{
				Query: &planpb.QueryPlanNode{
					Predicates: &planpb.Expr{
						Expr: &planpb.Expr_BinaryExpr{
							BinaryExpr: &planpb.BinaryExpr{
								Op: planpb.BinaryExpr_LogicalAnd,
								Left: &planpb.Expr{
									Expr: &planpb.Expr_TermExpr{
										TermExpr: &planpb.TermExpr{
											ColumnInfo: &planpb.ColumnInfo{
												IsPrimaryKey: true,
												DataType:     schemapb.DataType_Int64,
											},
											Values: []*planpb.GenericValue{
												{Val: &planpb.GenericValue_Int64Val{Int64Val: 1}},
												{Val: &planpb.GenericValue_Int64Val{Int64Val: 2}},
											},
										},
									},
								},
								Right: &planpb.Expr{
									Expr: &planpb.Expr_UnaryRangeExpr{
										UnaryRangeExpr: &planpb.UnaryRangeExpr{
											ColumnInfo: &planpb.ColumnInfo{
												IsPrimaryKey: false,
												DataType:     schemapb.DataType_Int64,
											},
											Op:    planpb.OpType_GreaterThan,
											Value: &planpb.GenericValue{Val: &planpb.GenericValue_Int64Val{Int64Val: 18}},
										},
									},
								},
							},
						},
					},
				},
			},
		}
		assert.Equal(t, common.PkFilterHasPkFilter, checkSegmentFilter(plan))
	})

	t.Run("OR with pk only on one side is not optimizable", func(t *testing.T) {
		plan := &planpb.PlanNode{
			Node: &planpb.PlanNode_Query{
				Query: &planpb.QueryPlanNode{
					Predicates: &planpb.Expr{
						Expr: &planpb.Expr_BinaryExpr{
							BinaryExpr: &planpb.BinaryExpr{
								Op: planpb.BinaryExpr_LogicalOr,
								Left: &planpb.Expr{
									Expr: &planpb.Expr_TermExpr{
										TermExpr: &planpb.TermExpr{
											ColumnInfo: &planpb.ColumnInfo{
												IsPrimaryKey: true,
												DataType:     schemapb.DataType_Int64,
											},
											Values: []*planpb.GenericValue{
												{Val: &planpb.GenericValue_Int64Val{Int64Val: 1}},
											},
										},
									},
								},
								Right: &planpb.Expr{
									Expr: &planpb.Expr_UnaryRangeExpr{
										UnaryRangeExpr: &planpb.UnaryRangeExpr{
											ColumnInfo: &planpb.ColumnInfo{
												IsPrimaryKey: false,
												DataType:     schemapb.DataType_Int64,
											},
											Op:    planpb.OpType_GreaterThan,
											Value: &planpb.GenericValue{Val: &planpb.GenericValue_Int64Val{Int64Val: 18}},
										},
									},
								},
							},
						},
					},
				},
			},
		}
		// OR(pk, non-pk) is not optimizable: the non-pk side is unconstrained,
		// so BF/min-max pruning cannot safely skip any segment.
		assert.Equal(t, common.PkFilterNoPkFilter, checkSegmentFilter(plan))
	})

	t.Run("NOT wrapping pk predicate returns no pk filter", func(t *testing.T) {
		plan := &planpb.PlanNode{
			Node: &planpb.PlanNode_Query{
				Query: &planpb.QueryPlanNode{
					Predicates: &planpb.Expr{
						Expr: &planpb.Expr_UnaryExpr{
							UnaryExpr: &planpb.UnaryExpr{
								Op: planpb.UnaryExpr_Not,
								Child: &planpb.Expr{
									Expr: &planpb.Expr_TermExpr{
										TermExpr: &planpb.TermExpr{
											ColumnInfo: &planpb.ColumnInfo{
												IsPrimaryKey: true,
												DataType:     schemapb.DataType_Int64,
											},
											Values: []*planpb.GenericValue{
												{Val: &planpb.GenericValue_Int64Val{Int64Val: 1}},
											},
										},
									},
								},
							},
						},
					},
				},
			},
		}
		// NOT negates the PK constraint; delegator's extractPkConstraint returns nil for NOT.
		// Return NoPkFilter to avoid spurious plan unmarshal with no optimization benefit.
		assert.Equal(t, common.PkFilterNoPkFilter, checkSegmentFilter(plan))
	})

	t.Run("varchar pk predicate", func(t *testing.T) {
		plan := &planpb.PlanNode{
			Node: &planpb.PlanNode_Query{
				Query: &planpb.QueryPlanNode{
					Predicates: &planpb.Expr{
						Expr: &planpb.Expr_TermExpr{
							TermExpr: &planpb.TermExpr{
								ColumnInfo: &planpb.ColumnInfo{
									IsPrimaryKey: true,
									DataType:     schemapb.DataType_VarChar,
								},
								Values: []*planpb.GenericValue{
									{Val: &planpb.GenericValue_StringVal{StringVal: "pk1"}},
								},
							},
						},
					},
				},
			},
		}
		assert.Equal(t, common.PkFilterHasPkFilter, checkSegmentFilter(plan))
	})
}

func TestAdvancedSearchAggregatesPKFilterHint(t *testing.T) {
	paramtable.Init()
	ctx := context.Background()
	schemaInfo := mustNewSchemaInfo(&schemapb.CollectionSchema{
		Name: "test_hybrid_pk_filter",
		Fields: []*schemapb.FieldSchema{
			{FieldID: 100, Name: "pk", DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
			{FieldID: 101, Name: "vec", DataType: schemapb.DataType_FloatVector, TypeParams: []*commonpb.KeyValuePair{{Key: common.DimKey, Value: "4"}}},
			{FieldID: 102, Name: "category", DataType: schemapb.DataType_VarChar},
		},
	})
	placeholderGroup, err := proto.Marshal(&commonpb.PlaceholderGroup{
		Placeholders: []*commonpb.PlaceholderValue{{
			Tag:    "$0",
			Type:   commonpb.PlaceholderType_FloatVector,
			Values: [][]byte{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}},
		}},
	})
	require.NoError(t, err)

	subRequest := func(expr string) *milvuspb.SubSearchRequest {
		return &milvuspb.SubSearchRequest{
			Dsl:              expr,
			Nq:               1,
			PlaceholderGroup: placeholderGroup,
			SearchParams: []*commonpb.KeyValuePair{
				{Key: common.MetricTypeKey, Value: "L2"},
				{Key: ParamsKey, Value: `{"nprobe": 10}`},
				{Key: AnnsFieldKey, Value: "vec"},
				{Key: TopKKey, Value: "10"},
			},
		}
	}

	for _, tc := range []struct {
		name string
		expr string
		want int32
	}{
		{name: "no filter", want: common.PkFilterNoPkFilter},
		{name: "pk filter in second sub-request", expr: "pk in [1, 2]", want: common.PkFilterHasPkFilter},
		{name: "non-pk filter only", expr: `category == "a"`, want: common.PkFilterNoPkFilter},
	} {
		t.Run(tc.name, func(t *testing.T) {
			task := &SearchTask{
				ctx:            ctx,
				collectionName: "test_hybrid_pk_filter",
				SearchRequest: &internalpb.SearchRequest{
					CollectionID: 1,
					PartitionIDs: []int64{1},
				},
				request: &milvuspb.SearchRequest{
					CollectionName: "test_hybrid_pk_filter",
					SearchParams: []*commonpb.KeyValuePair{
						{Key: LimitKey, Value: "10"},
					},
					SubReqs: []*milvuspb.SubSearchRequest{
						subRequest(""),
						subRequest(tc.expr),
					},
				},
				schema: schemaInfo,
				tr:     timerecord.NewTimeRecorder("test"),
			}

			require.NoError(t, task.initAdvancedSearchRequest(ctx))
			assert.Equal(t, tc.want, task.GetPkFilter())
		})
	}
}
