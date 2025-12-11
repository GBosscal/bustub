//===----------------------------------------------------------------------===//
//
//                         BusTub
//
// b_plus_tree.cpp
//
// Identification: src/storage/index/b_plus_tree.cpp
//
// Copyright (c) 2015-2025, Carnegie Mellon University Database Group
//
//===----------------------------------------------------------------------===//

#include "common/exception.h"
#include "storage/index/b_plus_tree.h"
#include "buffer/traced_buffer_pool_manager.h"
#include "storage/index/b_plus_tree_debug.h"
#include "storage/page/b_plus_tree_internal_page.h"
#include "storage/page/b_plus_tree_leaf_page.h"

namespace bustub {

FULL_INDEX_TEMPLATE_ARGUMENTS
BPLUSTREE_TYPE::BPlusTree(std::string name, page_id_t header_page_id, BufferPoolManager *buffer_pool_manager,
                          const KeyComparator &comparator, int leaf_max_size, int internal_max_size)
    : bpm_(std::make_shared<TracedBufferPoolManager>(buffer_pool_manager)),
      index_name_(std::move(name)),
      comparator_(std::move(comparator)),
      leaf_max_size_(leaf_max_size),
      internal_max_size_(internal_max_size),
      header_page_id_(header_page_id) {
  WritePageGuard guard = bpm_->WritePage(header_page_id_);
  auto root_page = guard.AsMut<BPlusTreeHeaderPage>();
  root_page->root_page_id_ = INVALID_PAGE_ID;
}

/**
 * @brief Helper function to decide whether current b+tree is empty
 * @return Returns true if this B+ tree has no keys and values.
 */
FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::IsEmpty() const -> bool { 
  ReadPageGuard guard = bpm_->ReadPage(header_page_id_);
  auto header_page = guard.As<BPlusTreeHeaderPage>();
  return header_page->root_page_id_ == INVALID_PAGE_ID;
}

/*****************************************************************************
 * SEARCH
 *****************************************************************************/
/**
 * @brief Return the only value that associated with input key
 *
 * This method is used for point query
 *
 * @param key input key
 * @param[out] result vector that stores the only value that associated with input key, if the value exists
 * @return : true means key exists
 */
FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::GetValue(const KeyType &key, std::vector<ValueType> *result) -> bool {
  result->clear();
  if (IsEmpty()) {
    return false;
  }

  Context ctx; // Declaration of context instance. Using the Context is not necessary but advised.
  ReadPageGuard header_guard = bpm_->ReadPage(header_page_id_);
  auto header_page = header_guard.As<BPlusTreeHeaderPage>();
  page_id_t root_id = header_page->root_page_id_;
  header_guard.Drop();

  ReadPageGuard root_guard = bpm_->ReadPage(root_id);
  auto current_page = root_guard.As<BPlusTreePage>();
  page_id_t current_id = root_id;

  while (!current_page->IsLeafPage()) {
    auto internal_page = root_guard.As<BPlusTreeInternalPage<KeyType, page_id_t, KeyComparator>>();
    page_id_t next_id = internal_page->Lookup(key, comparator_);
    root_guard.Drop();
    current_id = next_id;
    root_guard = bpm_->ReadPage(current_id);
    current_page = root_guard.As<BPlusTreePage>();
  }

  auto leaf_page = root_guard.As<LeafPage>();
  ValueType value;
  if (leaf_page->Lookup(key, &value, comparator_)) {
    result->push_back(value);
    return true;
  }
  return false;
}

/*****************************************************************************
 * INSERTION
 *****************************************************************************/
/**
 * @brief Insert constant key & value pair into b+ tree
 *
 * if current tree is empty, start new tree, update root page id and insert
 * entry; otherwise, insert into leaf page.
 *
 * @param key the key to insert
 * @param value the value associated with key
 * @return: since we only support unique key, if user try to insert duplicate
 * keys return false; otherwise, return true.
 */

namespace {
template <typename KeyType, typename ValueType, typename KeyComparator, ssize_t NumTombs>
auto SplitLeafPage(BPlusTree<KeyType, ValueType, KeyComparator, NumTombs> *tree,
                   typename BPlusTree<KeyType, ValueType, KeyComparator, NumTombs>::LeafPage *leaf,
                   BufferPoolManager *bpm) -> std::pair<KeyType, page_id_t> {
  auto new_leaf = bpm->NewPage<decltype(leaf)>();
  page_id_t new_leaf_id = new_leaf->GetPageId();
  int old_size = leaf->GetSize();
  int half = old_size / 2;

  new_leaf->Init(leaf->GetMaxSize());
  new_leaf->SetParentPageId(leaf->GetParentPageId());
  new_leaf->SetNextPageId(leaf->GetNextPageId());
  leaf->SetNextPageId(new_leaf_id);

  for (int i = half; i < old_size; ++i) {
    new_leaf->InsertAt(i - half, leaf->KeyAt(i), leaf->ValueAt(i));
  }
  leaf->SetSize(half);
  new_leaf->SetSize(old_size - half);

  return {new_leaf->KeyAt(0), new_leaf_id};
}

template <typename KeyType, typename ValueType, typename KeyComparator, ssize_t NumTombs>
auto SplitInternalPage(BPlusTree<KeyType, ValueType, KeyComparator, NumTombs> *tree,
                       typename BPlusTree<KeyType, ValueType, KeyComparator, NumTombs>::InternalPage *internal,
                       BufferPoolManager *bpm) -> std::pair<KeyType, page_id_t> {
  auto new_internal = bpm->NewPage<decltype(internal)>();
  page_id_t new_internal_id = new_internal->GetPageId();
  int old_size = internal->GetSize();
  int half = old_size / 2;
  KeyType split_key = internal->KeyAt(half);

  new_internal->Init(internal->GetMaxSize());
  new_internal->SetParentPageId(internal->GetParentPageId());

  new_internal->InsertAt(0, internal->KeyAt(half + 1), internal->ValueAt(half + 1));
  for (int i = half + 1; i < old_size; ++i) {
    new_internal->InsertAt(i - half, internal->KeyAt(i), internal->ValueAt(i));
    auto child_guard = bpm->WritePage(internal->ValueAt(i));
    auto child_page = child_guard.AsMut<BPlusTreePage>();
    child_page->SetParentPageId(new_internal_id);
  }

  internal->SetSize(half);
  new_internal->SetSize(old_size - half - 1);

  return {split_key, new_internal_id};
}
}  // namespace

FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::Insert(const KeyType &key, const ValueType &value) -> bool {
  // Declaration of context instance. Using the Context is not necessary but advised.
  Context ctx;

  WritePageGuard header_guard = bpm_->WritePage(header_page_id_);
  auto header_page = header_guard.AsMut<BPlusTreeHeaderPage>();
  page_id_t root_id = header_page->root_page_id_;

  if (root_id == INVALID_PAGE_ID) {
    auto root_leaf = bpm_->NewPage<LeafPage>();
    root_leaf->Init(leaf_max_size_);
    root_leaf->SetParentPageId(INVALID_PAGE_ID);
    root_leaf->SetNextPageId(INVALID_PAGE_ID);
    root_leaf->Insert(key, value, comparator_);
    header_page->root_page_id_ = root_leaf->GetPageId();
    ctx.header_page_ = std::move(header_guard);
    return true;
  }

  std::vector<WritePageGuard> path;
  WritePageGuard root_guard = bpm_->WritePage(root_id);
  auto current_page = root_guard.AsMut<BPlusTreePage>();
  page_id_t current_id = root_id;
  path.push_back(std::move(root_guard));

  while (!current_page->IsLeafPage()) {
    auto internal_page = path.back().AsMut<InternalPage>();
    page_id_t next_id = internal_page->Lookup(key, comparator_);
    path.back().Drop();
    WritePageGuard next_guard = bpm_->WritePage(next_id);
    current_page = next_guard.AsMut<BPlusTreePage>();
    path.push_back(std::move(next_guard));
  }

  auto leaf_page = path.back().AsMut<LeafPage>();
  if (leaf_page->Lookup(key, nullptr, comparator_)) {
    return false;
  }

  leaf_page->Insert(key, value, comparator_);
  page_id_t split_page_id = INVALID_PAGE_ID;
  KeyType split_key;
  bool split_occurred = false;

  if (leaf_page->GetSize() > leaf_page->GetMaxSize()) {
    auto [key, id] = SplitLeafPage(this, leaf_page, bpm_.get());
    split_key = key;
    split_page_id = id;
    split_occurred = true;
  }

  path.pop_back();
  page_id_t current_child_id = leaf_page->GetPageId();

  while (split_occurred && !path.empty()) {
    auto parent_guard = std::move(path.back());
    path.pop_back();
    auto parent_page = parent_guard.AsMut<InternalPage>();
    parent_page->InsertAt(parent_page->ValueIndex(current_child_id) + 1, split_key, split_page_id);

    if (parent_page->GetSize() > parent_page->GetMaxSize()) {
      auto [key, id] = SplitInternalPage(this, parent_page, bpm_.get());
      split_key = key;
      split_page_id = id;
      current_child_id = parent_page->GetPageId();
      split_occurred = true;
      path.push_back(std::move(parent_guard));
    } else {
      split_occurred = false;
    }
  }

  if (split_occurred) {
    auto new_root = bpm_->NewPage<InternalPage>();
    new_root->Init(internal_max_size_);
    new_root->SetParentPageId(INVALID_PAGE_ID);
    new_root->InsertAt(0, split_key, split_page_id);
    new_root->SetValueAt(0, root_id);

    auto old_root_guard = bpm_->WritePage(root_id);
    auto old_root = old_root_guard.AsMut<BPlusTreePage>();
    old_root->SetParentPageId(new_root->GetPageId());
    old_root_guard.Drop();

    auto new_split_guard = bpm_->WritePage(split_page_id);
    auto new_split_page = new_split_guard.AsMut<BPlusTreePage>();
    new_split_page->SetParentPageId(new_root->GetPageId());
    new_split_guard.Drop();

    header_page->root_page_id_ = new_root->GetPageId();
  }

  ctx.header_page_ = std::move(header_guard);
  return true;
}

/*****************************************************************************
 * REMOVE
 *****************************************************************************/
/**
 * @brief Delete key & value pair associated with input key
 * If current tree is empty, return immediately.
 * If not, User needs to first find the right leaf page as deletion target, then
 * delete entry from leaf page. Remember to deal with redistribute or merge if
 * necessary.
 *
 * @param key input key
 */

namespace {
template <typename KeyType, typename ValueType, typename KeyComparator, ssize_t NumTombs>
void CoalesceLeaves(BPlusTree<KeyType, ValueType, KeyComparator, NumTombs> *tree,
                    typename BPlusTree<KeyType, ValueType, KeyComparator, NumTombs>::LeafPage *left,
                    typename BPlusTree<KeyType, ValueType, KeyComparator, NumTombs>::LeafPage *right,
                    typename BPlusTree<KeyType, ValueType, KeyComparator, NumTombs>::InternalPage *parent,
                    int index, BufferPoolManager *bpm) {
  for (int i = 0; i < right->GetSize(); ++i) {
    left->Insert(right->KeyAt(i), right->ValueAt(i), tree->comparator_);
  }
  left->SetNextPageId(right->GetNextPageId());

  parent->RemoveAt(index);
  bpm->DeletePage(right->GetPageId());
}

template <typename KeyType, typename ValueType, typename KeyComparator, ssize_t NumTombs>
void CoalesceInternals(BPlusTree<KeyType, ValueType, KeyComparator, NumTombs> *tree,
                       typename BPlusTree<KeyType, ValueType, KeyComparator, NumTombs>::InternalPage *left,
                       typename BPlusTree<KeyType, ValueType, KeyComparator, NumTombs>::InternalPage *right,
                       typename BPlusTree<KeyType, ValueType, KeyComparator, NumTombs>::InternalPage *parent,
                       int index, BufferPoolManager *bpm) {
  KeyType middle_key = parent->KeyAt(index);
  left->Insert(middle_key, right->ValueAt(0));

  for (int i = 0; i < right->GetSize(); ++i) {
    left->Insert(right->KeyAt(i), right->ValueAt(i + 1));
    auto child_guard = bpm->WritePage(right->ValueAt(i + 1));
    child_guard.AsMut<BPlusTreePage>()->SetParentPageId(left->GetPageId());
  }

  parent->RemoveAt(index);
  bpm->DeletePage(right->GetPageId());
}

template <typename KeyType, typename ValueType, typename KeyComparator, ssize_t NumTombs>
bool RedistributeLeaf(typename BPlusTree<KeyType, ValueType, KeyComparator, NumTombs>::LeafPage *left,
                      typename BPlusTree<KeyType, ValueType, KeyComparator, NumTombs>::LeafPage *right,
                      typename BPlusTree<KeyType, ValueType, KeyComparator, NumTombs>::InternalPage *parent,
                      int index, const KeyComparator &comp) {
  if (left->GetSize() > left->GetMinSize()) {
    auto [key, val] = left->RemoveAndReturnLast();
    right->InsertAt(0, key, val);
    parent->SetKeyAt(index, right->KeyAt(0));
    return true;
  }
  if (right->GetSize() > right->GetMinSize()) {
    auto [key, val] = right->RemoveAndReturnFirst();
    left->Insert(key, val, comp);
    parent->SetKeyAt(index, right->KeyAt(0));
    return true;
  }
  return false;
}

template <typename KeyType, typename ValueType, typename KeyComparator, ssize_t NumTombs>
bool RedistributeInternal(typename BPlusTree<KeyType, ValueType, KeyComparator, NumTombs>::InternalPage *left,
                          typename BPlusTree<KeyType, ValueType, KeyComparator, NumTombs>::InternalPage *right,
                          typename BPlusTree<KeyType, ValueType, KeyComparator, NumTombs>::InternalPage *parent,
                          int index, BufferPoolManager *bpm) {
  if (left->GetSize() > left->GetMinSize()) {
    KeyType key = left->KeyAt(left->GetSize() - 1);
    page_id_t val = left->ValueAt(left->GetSize());
    left->RemoveAt(left->GetSize() - 1);

    right->InsertAt(0, parent->KeyAt(index), right->ValueAt(0));
    right->SetValueAt(0, val);
    parent->SetKeyAt(index, key);

    auto child_guard = bpm->WritePage(val);
    child_guard.AsMut<BPlusTreePage>()->SetParentPageId(right->GetPageId());
    return true;
  }
  if (right->GetSize() > right->GetMinSize()) {
    KeyType key = right->KeyAt(0);
    page_id_t val = right->ValueAt(0);
    right->RemoveAt(0);

    left->Insert(left->KeyAt(left->GetSize() - 1), val);
    parent->SetKeyAt(index, key);

    auto child_guard = bpm->WritePage(val);
    child_guard.AsMut<BPlusTreePage>()->SetParentPageId(left->GetPageId());
    return true;
  }
  return false;
}
}  // namespace

FULL_INDEX_TEMPLATE_ARGUMENTS
void BPLUSTREE_TYPE::Remove(const KeyType &key) {
  if (IsEmpty()) {
    return;
  }

  // Declaration of context instance.
  Context ctx;
  WritePageGuard header_guard = bpm_->WritePage(header_page_id_);
  auto header_page = header_guard.AsMut<BPlusTreeHeaderPage>();
  page_id_t root_id = header_page->root_page_id_;

  std::vector<WritePageGuard> path;
  WritePageGuard root_guard = bpm_->WritePage(root_id);
  auto current_page = root_guard.AsMut<BPlusTreePage>();
  path.push_back(std::move(root_guard));

  while (!current_page->IsLeafPage()) {
    auto internal_page = path.back().AsMut<InternalPage>();
    page_id_t next_id = internal_page->Lookup(key, comparator_);
    path.back().Drop();
    WritePageGuard next_guard = bpm_->WritePage(next_id);
    current_page = next_guard.AsMut<BPlusTreePage>();
    path.push_back(std::move(next_guard));
  }

  auto leaf_page = path.back().AsMut<LeafPage>();
  if (!leaf_page->Lookup(key, nullptr, comparator_)) {
    return;
  }

  leaf_page->Remove(key, comparator_);
  bool underflow = leaf_page->GetSize() < leaf_page->GetMinSize();
  page_id_t current_id = leaf_page->GetPageId();
  path.pop_back();

  while (underflow && !path.empty()) {
    auto parent_guard = std::move(path.back());
    path.pop_back();
    auto parent_page = parent_guard.AsMut<InternalPage>();
    int index = parent_page->ValueIndex(current_id);
    bool is_left = (index == 0);
    page_id_t sibling_id = is_left ? parent_page->ValueAt(1) : parent_page->ValueAt(index - 1);
    int sibling_index = is_left ? 1 : index - 1;

    WritePageGuard sibling_guard = bpm_->WritePage(sibling_id);
    bool resolved = false;

    if (leaf_page->IsLeafPage()) {
      auto sibling_leaf = sibling_guard.AsMut<LeafPage>();
      if (RedistributeLeaf(leaf_page, sibling_leaf, parent_page, is_left ? 0 : index - 1, comparator_)) {
        resolved = true;
      } else {
        if (is_left) {
          CoalesceLeaves(this, sibling_leaf, leaf_page, parent_page, 0, bpm_.get());
          current_id = sibling_id;
        } else {
          CoalesceLeaves(this, leaf_page, sibling_leaf, parent_page, index - 1, bpm_.get());
        }
      }
    } else {
      auto sibling_internal = sibling_guard.AsMut<InternalPage>();
      if (RedistributeInternal(leaf_page, sibling_internal, parent_page, index, bpm_.get())) {
        resolved = true;
      } else {
        if (is_left) {
          CoalesceInternals(this, sibling_internal, leaf_page, parent_page, 0, bpm_.get());
          current_id = sibling_id;
        } else {
          CoalesceInternals(this, leaf_page, sibling_internal, parent_page, index - 1, bpm_.get());
        }
      }
    }

    sibling_guard.Drop();
    underflow = !resolved && (parent_page->GetSize() < parent_page->GetMinSize());
    if (!underflow) {
      path.push_back(std::move(parent_guard));
    } else {
      current_id = parent_page->GetPageId();
    }
  }

  if (underflow) {
    if (path.empty()) {
      auto root_guard = bpm_->WritePage(root_id);
      auto root_page = root_guard.AsMut<BPlusTreePage>();
      if (root_page->GetSize() == 0) {
        header_page->root_page_id_ = INVALID_PAGE_ID;
        bpm_->DeletePage(root_id);
      } else if (!root_page->IsLeafPage()) {
        auto root_internal = root_guard.AsMut<InternalPage>();
        page_id_t new_root_id = root_internal->ValueAt(0);
        auto new_root_guard = bpm_->WritePage(new_root_id);
        new_root_guard.AsMut<BPlusTreePage>()->SetParentPageId(INVALID_PAGE_ID);
        header_page->root_page_id_ = new_root_id;
        bpm_->DeletePage(root_id);
      }
    }
  }

  ctx.header_page_ = std::move(header_guard);
}

/*****************************************************************************
 * INDEX ITERATOR
 *****************************************************************************/
/**
 * @brief Input parameter is void, find the leftmost leaf page first, then construct
 * index iterator
 *
 * You may want to implement this while implementing Task #3.
 *
 * @return : index iterator
 */
FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::Begin() -> INDEXITERATOR_TYPE {
  if (IsEmpty()) {
    return End();
  }

  ReadPageGuard header_guard = bpm_->ReadPage(header_page_id_);
  auto header_page = header_guard.As<BPlusTreeHeaderPage>();
  page_id_t root_id = header_page->root_page_id_;
  header_guard.Drop();

  ReadPageGuard current_guard = bpm_->ReadPage(root_id);
  auto current_page = current_guard.As<BPlusTreePage>();

  while (!current_page->IsLeafPage()) {
    auto internal_page = current_guard.As<InternalPage>();
    page_id_t next_id = internal_page->ValueAt(0);
    current_guard.Drop();
    current_guard = bpm_->ReadPage(next_id);
    current_page = current_guard.As<BPlusTreePage>();
  }

  auto leaf_page = current_guard.As<LeafPage>();
  return INDEXITERATOR_TYPE(leaf_page->GetPageId(), 0, bpm_);
}

/**
 * @brief Input parameter is low key, find the leaf page that contains the input key
 * first, then construct index iterator
 * @return : index iterator
 */
FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::Begin(const KeyType &key) -> INDEXITERATOR_TYPE {
  if (IsEmpty()) {
    return End();
  }

  ReadPageGuard header_guard = bpm_->ReadPage(header_page_id_);
  auto header_page = header_guard.As<BPlusTreeHeaderPage>();
  page_id_t root_id = header_page->root_page_id_;
  header_guard.Drop();

  ReadPageGuard current_guard = bpm_->ReadPage(root_id);
  auto current_page = current_guard.As<BPlusTreePage>();

  while (!current_page->IsLeafPage()) {
    auto internal_page = current_guard.As<InternalPage>();
    page_id_t next_id = internal_page->Lookup(key, comparator_);
    current_guard.Drop();
    current_guard = bpm_->ReadPage(next_id);
    current_page = current_guard.As<BPlusTreePage>();
  }

  auto leaf_page = current_guard.As<LeafPage>();
  int pos = leaf_page->KeyIndex(key, comparator_);
  if (pos == leaf_page->GetSize() || comparator_(key, leaf_page->KeyAt(pos)) < 0) {
    pos = 0;
  }

  return INDEXITERATOR_TYPE(leaf_page->GetPageId(), pos, bpm_);
}

/**
 * @brief Input parameter is void, construct an index iterator representing the end
 * of the key/value pair in the leaf node
 * @return : index iterator
 */
FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::End() -> INDEXITERATOR_TYPE { 
  return INDEXITERATOR_TYPE(INVALID_PAGE_ID, 0, bpm_); 
}

/**
 * @return Page id of the root of this tree
 *
 * You may want to implement this while implementing Task #3.
 */
FULL_INDEX_TEMPLATE_ARGUMENTS
auto BPLUSTREE_TYPE::GetRootPageId() -> page_id_t { 
  ReadPageGuard guard = bpm_->ReadPage(header_page_id_);
  return guard.As<BPlusTreeHeaderPage>()->root_page_id_;
}

template class BPlusTree<GenericKey<4>, RID, GenericComparator<4>>;

template class BPlusTree<GenericKey<8>, RID, GenericComparator<8>>;
template class BPlusTree<GenericKey<8>, RID, GenericComparator<8>, 3>;
template class BPlusTree<GenericKey<8>, RID, GenericComparator<8>, 2>;
template class BPlusTree<GenericKey<8>, RID, GenericComparator<8>, 1>;
template class BPlusTree<GenericKey<8>, RID, GenericComparator<8>, -1>;

template class BPlusTree<GenericKey<16>, RID, GenericComparator<16>>;

template class BPlusTree<GenericKey<32>, RID, GenericComparator<32>>;

template class BPlusTree<GenericKey<64>, RID, GenericComparator<64>>;

}  // namespace bustub
