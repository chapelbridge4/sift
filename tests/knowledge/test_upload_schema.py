from app.models.schemas import UploadFilesRequest


def test_upload_files_request_make_knowledge_defaults():
    req = UploadFilesRequest(collection_name="c", file_paths=["a.pdf"])
    assert req.make_knowledge is False
    assert req.knowledge_profile == "papers"
    assert req.knowledge_model is None


def test_upload_files_request_make_knowledge_flags():
    req = UploadFilesRequest(
        collection_name="c",
        file_paths=["a.pdf"],
        make_knowledge=True,
        knowledge_profile="papers",
        knowledge_model="/tmp/custom.gguf",
    )
    assert req.make_knowledge is True
    assert req.knowledge_model == "/tmp/custom.gguf"