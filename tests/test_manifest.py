from miia.data.manifest import DatasetRecord, PhashIndex, build_ret3_train


def record(dataset, image_id, split, phash):
    return DatasetRecord(
        dataset=dataset,
        image_id=image_id,
        image_path=__file__,
        captions=[str(i) for i in range(5)],
        split=split,
        sha256=image_id,
        phash=phash,
        sources=[f"{dataset}:{image_id}"],
    )


def test_phash_bktree_threshold():
    index = PhashIndex()
    item = record("rsicd", "a", "test", "0000000000000000")
    index.add(item)
    assert index.search("0000000000000003", threshold=2) == [item]
    assert index.search("0000000000000007", threshold=2) == []


def test_test_split_wins_over_train_duplicate():
    train = record("rsicd", "train", "train", "0000000000000000")
    test = record("rsitmd", "test", "test", "0000000000000001")
    kept, audit = build_ret3_train([train, test], threshold=2)
    assert kept == []
    assert audit["excluded_records"] == 1
    assert train.exclusion_reason == "near_duplicate_of_held_out"


def test_cross_dataset_train_overlap_is_deduplicated():
    left = record("rsicd", "left", "train", "0011223344556677")
    right = record("rsitmd", "right", "train", "0011223344556677")
    kept, audit = build_ret3_train([left, right])
    assert kept == [left]
    assert audit["kept_train_records"] == 1
    assert audit["excluded_records"] == 1
    assert audit["audited_overlaps"] == 1
    assert audit["exclusions"][0]["reason"] == "near_duplicate_in_train"


def test_no_kept_record_overlaps_any_dataset_test_split():
    train = record("rsicd", "train", "train", "abcdef0000000000")
    other_test = record("ucm", "test", "test", "abcdef0000000003")
    unrelated = record("rsitmd", "clean", "train", "1234567890abcdef")
    kept, _ = build_ret3_train([train, other_test, unrelated], threshold=2)
    assert kept == [unrelated]
