package delegator

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestCloneSnapshotItemsIsolatesSubSearches(t *testing.T) {
	original := []SnapshotItem{
		{
			NodeID: 1,
			Segments: []SegmentEntry{
				{NodeID: 1, SegmentID: 10, PartitionID: 100},
				{NodeID: 1, SegmentID: 11, PartitionID: 100},
			},
		},
		{
			NodeID: 2,
			Segments: []SegmentEntry{
				{NodeID: 2, SegmentID: 20, PartitionID: 100},
			},
		},
	}

	first := cloneSnapshotItems(original)
	second := cloneSnapshotItems(original)
	require.NotSame(t, original[0].Segments, first[0].Segments)
	require.NotSame(t, first[0].Segments, second[0].Segments)

	first[0].NodeID = 99
	first[0].Segments[0].SegmentID = 1000
	first[1].Segments = nil

	require.Equal(t, int64(1), original[0].NodeID)
	require.Equal(t, int64(10), original[0].Segments[0].SegmentID)
	require.Len(t, original[1].Segments, 1)
	require.Equal(t, int64(1), second[0].NodeID)
	require.Equal(t, int64(10), second[0].Segments[0].SegmentID)
	require.Len(t, second[1].Segments, 1)
	require.Nil(t, cloneSnapshotItems(nil))
}
