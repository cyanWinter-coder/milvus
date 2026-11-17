package hellomilvus

import (
	"context"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"time"

	"google.golang.org/protobuf/proto"

	"github.com/milvus-io/milvus-proto/go-api/v3/commonpb"
	"github.com/milvus-io/milvus-proto/go-api/v3/milvuspb"
	"github.com/milvus-io/milvus-proto/go-api/v3/schemapb"
	"github.com/milvus-io/milvus/internal/proxy"
	"github.com/milvus-io/milvus/pkg/v3/common"
	"github.com/milvus-io/milvus/pkg/v3/util/funcutil"
	"github.com/milvus-io/milvus/pkg/v3/util/merr"
	"github.com/milvus-io/milvus/pkg/v3/util/metric"
	"github.com/milvus-io/milvus/tests/integration"
)

const (
	hybridFilterPKField       = "id"
	hybridFilterCategoryField = "category"
	hybridFilterVecAField     = "vec_a"
	hybridFilterVecBField     = "vec_b"
)

// TestHybridSearchFilterPushdown covers the filter/pagination/delete/consistency
// contract of hybrid search. The query includes both a PK range and a scalar
// predicate so the Proxy must retain the PK hint for QueryNode segment pruning
// while every sub-search applies its own predicate.
func (s *HelloMilvusSuite) TestHybridSearchFilterPushdown() {
	c := s.Cluster
	ctx, cancel := context.WithCancel(c.GetContext())
	defer cancel()

	const (
		dim    = 4
		rowNum = 60
	)
	collectionName := "TestHybridFilter" + funcutil.GenRandomStr()
	filterExpr := fmt.Sprintf(`%s == "a" and %s >= 8 and %s < 40`,
		hybridFilterCategoryField, hybridFilterPKField, hybridFilterPKField)
	emptyExpr := fmt.Sprintf(`%s == "missing"`, hybridFilterCategoryField)

	schema := integration.ConstructSchema(collectionName, dim, false,
		&schemapb.FieldSchema{FieldID: 100, Name: hybridFilterPKField, DataType: schemapb.DataType_Int64, IsPrimaryKey: true},
		&schemapb.FieldSchema{FieldID: 101, Name: hybridFilterCategoryField, DataType: schemapb.DataType_VarChar, TypeParams: []*commonpb.KeyValuePair{{Key: common.MaxLengthKey, Value: "64"}}},
		&schemapb.FieldSchema{FieldID: 102, Name: hybridFilterVecAField, DataType: schemapb.DataType_FloatVector, TypeParams: []*commonpb.KeyValuePair{{Key: common.DimKey, Value: strconv.Itoa(dim)}}},
		&schemapb.FieldSchema{FieldID: 103, Name: hybridFilterVecBField, DataType: schemapb.DataType_FloatVector, TypeParams: []*commonpb.KeyValuePair{{Key: common.DimKey, Value: strconv.Itoa(dim)}}},
	)
	marshaledSchema, err := proto.Marshal(schema)
	s.Require().NoError(err)
	createResp, err := c.MilvusClient.CreateCollection(ctx, &milvuspb.CreateCollectionRequest{
		CollectionName: collectionName,
		Schema:         marshaledSchema,
		ShardsNum:      common.DefaultShardsNum,
	})
	s.Require().NoError(merr.CheckRPCCall(createResp, err))

	pks := integration.NewInt64FieldDataWithStart(hybridFilterPKField, rowNum, 0)
	categories := make([]string, rowNum)
	vecA := make([]float32, rowNum*dim)
	vecB := make([]float32, rowNum*dim)
	for i := 0; i < rowNum; i++ {
		switch {
		case i < 20:
			categories[i] = "a"
		case i < 40:
			categories[i] = "b"
		default:
			categories[i] = "c"
		}
		vecA[i*dim] = float32(i)
		vecB[i*dim] = float32(rowNum - i)
	}
	categoryData := &schemapb.FieldData{
		Type:      schemapb.DataType_VarChar,
		FieldName: hybridFilterCategoryField,
		Field: &schemapb.FieldData_Scalars{Scalars: &schemapb.ScalarField{
			Data: &schemapb.ScalarField_StringData{StringData: &schemapb.StringArray{Data: categories}},
		}},
	}
	insertResp, err := c.MilvusClient.Insert(ctx, &milvuspb.InsertRequest{
		CollectionName: collectionName,
		FieldsData: []*schemapb.FieldData{
			pks,
			categoryData,
			floatVectorFieldData(hybridFilterVecAField, vecA, dim),
			floatVectorFieldData(hybridFilterVecBField, vecB, dim),
		},
		HashKeys: integration.GenerateHashKeys(rowNum),
		NumRows:  rowNum,
	})
	s.Require().NoError(merr.CheckRPCCall(insertResp, err))

	flushResp, err := c.MilvusClient.Flush(ctx, &milvuspb.FlushRequest{CollectionNames: []string{collectionName}})
	s.Require().NoError(err)
	flushTs := flushResp.GetCollFlushTs()[collectionName]
	s.Require().NotEmpty(flushResp.GetFlushCollSegIDs()[collectionName].GetData())
	s.WaitForFlush(ctx, flushResp.GetFlushCollSegIDs()[collectionName].GetData(), flushTs, "", collectionName)

	for _, field := range []string{hybridFilterVecAField, hybridFilterVecBField} {
		indexResp, err := c.MilvusClient.CreateIndex(ctx, &milvuspb.CreateIndexRequest{
			CollectionName: collectionName,
			FieldName:      field,
			IndexName:      field + "_idx",
			ExtraParams:    integration.ConstructIndexParam(dim, integration.IndexFaissIDMap, metric.L2),
		})
		s.Require().NoError(merr.CheckRPCCall(indexResp, err))
		s.WaitForIndexBuiltWithIndexName(ctx, collectionName, field, field+"_idx")
	}
	categoryIndex, err := c.MilvusClient.CreateIndex(ctx, &milvuspb.CreateIndexRequest{
		CollectionName: collectionName,
		FieldName:      hybridFilterCategoryField,
		IndexName:      hybridFilterCategoryField + "_idx",
		ExtraParams:    []*commonpb.KeyValuePair{{Key: common.IndexTypeKey, Value: "INVERTED"}},
	})
	s.Require().NoError(merr.CheckRPCCall(categoryIndex, err))
	s.WaitForIndexBuiltWithIndexName(ctx, collectionName, hybridFilterCategoryField, hybridFilterCategoryField+"_idx")

	loadResp, err := c.MilvusClient.LoadCollection(ctx, &milvuspb.LoadCollectionRequest{CollectionName: collectionName})
	s.Require().NoError(merr.CheckRPCCall(loadResp, err))
	s.WaitForLoad(ctx, collectionName)

	baseIDs := s.int64QueryIDs(ctx, collectionName, filterExpr, commonpb.ConsistencyLevel_Strong)
	s.Require().ElementsMatch([]int64{8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19}, baseIDs)

	fullIDs := s.hybridFilterIDs(ctx, collectionName, filterExpr, 12, 0, commonpb.ConsistencyLevel_Strong)
	s.Require().Len(fullIDs, 12)
	s.Require().NoError(requireUniqueIDs(fullIDs))
	s.Require().Subset(baseIDs, fullIDs)

	page0 := s.hybridFilterIDs(ctx, collectionName, filterExpr, 6, 0, commonpb.ConsistencyLevel_Strong)
	page1 := s.hybridFilterIDs(ctx, collectionName, filterExpr, 6, 6, commonpb.ConsistencyLevel_Strong)
	s.Require().ElementsMatch(fullIDs, append(append([]int64(nil), page0...), page1...))
	s.Require().NoError(requireUniqueIDs(append(append([]int64(nil), page0...), page1...)))

	deleteResp, err := c.MilvusClient.Delete(ctx, &milvuspb.DeleteRequest{
		CollectionName: collectionName,
		Expr:           fmt.Sprintf("%s in [8, 9]", hybridFilterPKField),
	})
	s.Require().NoError(merr.CheckRPCCall(deleteResp, err))
	s.Require().Eventually(func() bool {
		ids := s.int64QueryIDs(ctx, collectionName, filterExpr, commonpb.ConsistencyLevel_Strong)
		return !containsID(ids, 8) && !containsID(ids, 9)
	}, 10*time.Second, 100*time.Millisecond)

	afterDeleteIDs := s.int64QueryIDs(ctx, collectionName, filterExpr, commonpb.ConsistencyLevel_Strong)
	s.Require().Len(afterDeleteIDs, 10)
	strongIDs := s.hybridFilterIDs(ctx, collectionName, filterExpr, 12, 0, commonpb.ConsistencyLevel_Strong)
	s.Require().NotContains(strongIDs, int64(8))
	s.Require().NotContains(strongIDs, int64(9))
	s.Require().Subset(afterDeleteIDs, strongIDs)
	s.Require().NoError(requireUniqueIDs(strongIDs))

	for _, level := range []commonpb.ConsistencyLevel{
		commonpb.ConsistencyLevel_Strong,
		commonpb.ConsistencyLevel_Bounded,
		commonpb.ConsistencyLevel_Eventually,
	} {
		ids := s.hybridFilterIDs(ctx, collectionName, filterExpr, 12, 0, level)
		s.Require().NoError(requireUniqueIDs(ids))
		s.Require().Subset(baseIDs, ids)
	}

	emptyIDs := s.hybridFilterIDs(ctx, collectionName, emptyExpr, 10, 0, commonpb.ConsistencyLevel_Strong)
	s.Require().Empty(emptyIDs)
}

func floatVectorFieldData(fieldName string, values []float32, dim int) *schemapb.FieldData {
	return &schemapb.FieldData{
		Type:      schemapb.DataType_FloatVector,
		FieldName: fieldName,
		Field: &schemapb.FieldData_Vectors{Vectors: &schemapb.VectorField{
			Dim:  int64(dim),
			Data: &schemapb.VectorField_FloatVector{FloatVector: &schemapb.FloatArray{Data: values}},
		}},
	}
}

func (s *HelloMilvusSuite) hybridFilterIDs(
	ctx context.Context,
	collectionName string,
	expr string,
	limit, offset int,
	level commonpb.ConsistencyLevel,
) []int64 {
	// Every page over-fetches beyond the tested window so fusion sees the same
	// candidate set as the full-result reference query.
	const subTopK = 32
	vecA := s.filteredSubSearch(collectionName, hybridFilterVecAField, expr, []float32{10, 0, 0, 0}, subTopK)
	vecB := s.filteredSubSearch(collectionName, hybridFilterVecBField, expr, []float32{50, 0, 0, 0}, subTopK)
	rankJSON, err := json.Marshal(map[string]float64{proxy.RRFParamsKey: 60})
	s.Require().NoError(err)
	resp, err := s.Cluster.MilvusClient.HybridSearch(ctx, &milvuspb.HybridSearchRequest{
		CollectionName:        collectionName,
		Requests:              []*milvuspb.SearchRequest{vecA, vecB},
		RankParams:            hybridRankParams(string(rankJSON), limit, offset),
		OutputFields:          []string{hybridFilterPKField, hybridFilterCategoryField},
		ConsistencyLevel:      level,
		UseDefaultConsistency: false,
	})
	s.Require().NoError(merr.CheckRPCCall(resp, err))
	ids := resp.GetResults().GetIds().GetIntId().GetData()
	if len(ids) == 0 {
		for _, topk := range resp.GetResults().GetTopks() {
			s.Require().Zero(topk)
		}
	}
	return ids
}

func (s *HelloMilvusSuite) int64QueryIDs(
	ctx context.Context,
	collectionName, expr string,
	level commonpb.ConsistencyLevel,
) []int64 {
	resp, err := s.Cluster.MilvusClient.Query(ctx, &milvuspb.QueryRequest{
		CollectionName:        collectionName,
		Expr:                  expr,
		OutputFields:          []string{hybridFilterPKField},
		ConsistencyLevel:      level,
		UseDefaultConsistency: false,
	})
	s.Require().NoError(merr.CheckRPCCall(resp, err))
	for _, field := range resp.GetOutputFields() {
		if field.GetFieldName() == hybridFilterPKField {
			return field.GetScalars().GetLongData().GetData()
		}
	}
	return nil
}

func (s *HelloMilvusSuite) filteredSubSearch(
	collectionName, fieldName, expr string,
	query []float32,
	topk int,
) *milvuspb.SearchRequest {
	req := integration.ConstructSearchRequest(
		"", collectionName, expr, fieldName, schemapb.DataType_FloatVector,
		nil, metric.L2, integration.GetSearchParams(integration.IndexFaissIDMap, metric.L2),
		1, len(query), topk, -1,
	)
	placeholder := &commonpb.PlaceholderGroup{}
	s.Require().NoError(proto.Unmarshal(req.GetPlaceholderGroup(), placeholder))
	raw := make([]byte, len(query)*4)
	for i, value := range query {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(value))
	}
	placeholder.Placeholders[0].Values[0] = raw
	encoded, err := proto.Marshal(placeholder)
	s.Require().NoError(err)
	req.SearchInput = &milvuspb.SearchRequest_PlaceholderGroup{PlaceholderGroup: encoded}
	req.Nq = 1
	return req
}

func hybridRankParams(rrfJSON string, limit, offset int) []*commonpb.KeyValuePair {
	return []*commonpb.KeyValuePair{
		{Key: proxy.RankTypeKey, Value: "rrf"},
		{Key: proxy.ParamsKey, Value: rrfJSON},
		{Key: proxy.LimitKey, Value: strconv.Itoa(limit)},
		{Key: proxy.OffsetKey, Value: strconv.Itoa(offset)},
		{Key: proxy.RoundDecimalKey, Value: "-1"},
	}
}

func requireUniqueIDs(ids []int64) error {
	seen := make(map[int64]struct{}, len(ids))
	for _, id := range ids {
		if _, ok := seen[id]; ok {
			return fmt.Errorf("duplicate id %d in result %v", id, ids)
		}
		seen[id] = struct{}{}
	}
	return nil
}

func containsID(ids []int64, target int64) bool {
	for _, id := range ids {
		if id == target {
			return true
		}
	}
	return false
}
