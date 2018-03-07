import trcdproc.reorganize.common as reorg
from trcdproc.core import H5File


def test_recursive_copy(organized_clean_data: H5File, starts_empty: H5File):
    """Ensures that the copy operation copies everything
    """
    reorg.recursive_copy(organized_clean_data, starts_empty)
    src_items = []
    organized_clean_data.visit(lambda x: src_items.append(x))
    dest_items = []
    starts_empty.visit(lambda x: dest_items.append(x))
    src_items = set(src_items)
    dest_items = set(src_items)
    assert src_items == dest_items
