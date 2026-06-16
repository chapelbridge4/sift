"""
FastAPI application for Brain-inspired RAG system.

Provides REST API endpoints for:
- Collection management (build, delete)
- Document upload and indexing
- Query and retrieval with LLM generation
"""

import sys
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.brain import PrefrontalCortex
from app.config import get_settings
from app.models.schemas import (
    CollectionCreate,
    CollectionDelete,
    CollectionResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    RetrievedDocument,
    UploadFilesRequest,
    UploadFilesResponse,
)
from app.security import UnsafePathError, resolve_safe_paths

# Configure logging
settings = get_settings()
logger.remove()
logger.add(
    sys.stderr,
    format=settings.LOG_FORMAT,
    level=settings.LOG_LEVEL
)
logger.add(
    "logs/brain_rag.log",
    rotation="10 MB",
    retention="7 days",
    format=settings.LOG_FORMAT,
    level=settings.LOG_LEVEL
)


# Global brain instance
brain: PrefrontalCortex = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown events."""
    # Startup
    global brain
    logger.info("Starting Brain-inspired RAG system...")

    brain = PrefrontalCortex()
    await brain.initialize()

    logger.info("Brain systems initialized and ready")

    yield

    # Shutdown
    logger.info("Shutting down Brain-inspired RAG system...")
    # Cleanup if needed


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Brain-inspired RAG system with Qdrant and MLX for Apple Silicon",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allow_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint."""
    return {
        "message": "Brain-inspired RAG API",
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint.

    Verifies:
    - API is running
    - Qdrant connection
    - MLX model loading
    """
    logger.info("Health check requested")

    qdrant_ok = await brain.hippocampus.qdrant_service.health_check()
    mlx_ok = await brain.llm_service.health_check()

    return HealthResponse(
        status="healthy" if (qdrant_ok and mlx_ok) else "degraded",
        app_name=settings.APP_NAME,
        version=settings.APP_VERSION,
        qdrant_connected=qdrant_ok,
        mlx_connected=mlx_ok
    )


@app.post("/build_collection", response_model=CollectionResponse, tags=["Collections"])
async def build_collection(request: CollectionCreate):
    """
    Create a new collection for document storage.

    This creates a memory space in the Hippocampus with hybrid search support.

    Args:
        request: Collection creation parameters

    Returns:
        Collection creation result
    """
    logger.info(f"Building collection: {request.collection_name}")

    try:
        # Check if collection already exists
        exists = await brain.hippocampus.memory_exists(request.collection_name)

        if exists:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Collection '{request.collection_name}' already exists"
            )

        # Create collection
        success = await brain.hippocampus.create_memory_space(
            collection_name=request.collection_name
        )

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create collection"
            )

        return CollectionResponse(
            collection_name=request.collection_name,
            status="created",
            vectors_count=0,
            message=f"Collection '{request.collection_name}' created successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building collection: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating collection: {str(e)}"
        )


@app.post("/upload_files", response_model=UploadFilesResponse, tags=["Documents"])
async def upload_files(request: UploadFilesRequest):
    """
    Upload and index documents to a collection.

    This forms new memories in the Hippocampus by:
    1. Parsing documents into chunks
    2. Generating hybrid embeddings (dense + sparse)
    3. Storing in Qdrant with metadata

    Args:
        request: Upload request with file paths and collection name

    Returns:
        Upload result with statistics
    """
    logger.info(
        f"Uploading {len(request.file_paths)} files to collection '{request.collection_name}'"
    )

    start_time = time.time()

    try:
        # Check if collection exists
        exists = await brain.hippocampus.memory_exists(request.collection_name)

        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{request.collection_name}' does not exist. Create it first."
            )

        # Reject paths that escape the corpus sandbox before touching disk
        try:
            safe_paths = resolve_safe_paths(request.file_paths, settings.ALLOWED_CORPUS_DIR)
        except UnsafePathError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

        # Form memories from documents
        result = await brain.hippocampus.form_memories(
            collection_name=request.collection_name,
            file_paths=safe_paths,
            batch_size=request.batch_size or 32
        )

        processing_time = time.time() - start_time

        if not result["success"]:
            return UploadFilesResponse(
                collection_name=request.collection_name,
                processed_files=0,
                total_chunks=0,
                failed_files=request.file_paths,
                processing_time_seconds=processing_time,
                message=result.get("message", "Failed to process files")
            )

        return UploadFilesResponse(
            collection_name=request.collection_name,
            processed_files=result.get("processed_files", len(request.file_paths)),
            total_chunks=result.get("total_chunks", 0),
            failed_files=[],
            processing_time_seconds=processing_time,
            message=f"Successfully indexed {result['total_chunks']} chunks"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading files: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error uploading files: {str(e)}"
        )


@app.delete("/delete_collection", response_model=CollectionResponse, tags=["Collections"])
async def delete_collection(request: CollectionDelete):
    """
    Delete a collection and all its documents.

    This removes a memory space from the Hippocampus.

    Args:
        request: Delete request with collection name

    Returns:
        Deletion result
    """
    logger.info(f"Deleting collection: {request.collection_name}")

    try:
        # Check if collection exists
        exists = await brain.hippocampus.memory_exists(request.collection_name)

        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{request.collection_name}' does not exist"
            )

        # Delete collection
        success = await brain.hippocampus.forget_memories(request.collection_name)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete collection"
            )

        return CollectionResponse(
            collection_name=request.collection_name,
            status="deleted",
            vectors_count=0,
            message=f"Collection '{request.collection_name}' deleted successfully"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting collection: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting collection: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse, tags=["Query"])
async def query(request: QueryRequest):
    """
    Query the RAG system with hybrid search and LLM generation.

    This orchestrates the full brain pipeline:
    1. Hippocampus: Retrieve relevant memories (hybrid search)
    2. Amygdala: Rank by importance
    3. Working Memory: Maintain conversation context
    4. Prefrontal Cortex: Generate reasoned response

    Args:
        request: Query request parameters

    Returns:
        Query response with answer and retrieved documents
    """
    logger.info(
        "Query request received "
        f"(collection='{request.collection_name}', query_length={len(request.query)}, top_k={request.top_k})"
    )

    start_time = time.time()

    try:
        # Check if collection exists
        exists = await brain.hippocampus.memory_exists(request.collection_name)

        if not exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Collection '{request.collection_name}' does not exist"
            )

        # Validate model_profile if provided
        effective_model_profile = None
        if request.model_profile:
            available_profiles = list(settings.MODEL_PROFILES.keys())
            if request.model_profile.value not in available_profiles:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Unknown model_profile '{request.model_profile.value}'. Available: {available_profiles}"
                )
            effective_model_profile = request.model_profile.value

        # Determine effective fusion method
        effective_fusion = request.fusion_method.value if request.fusion_method else "rrf"

        # Execute reasoning with full brain pipeline
        result = await brain.reason_with_context(
            query=request.query,
            collection_name=request.collection_name,
            top_k=request.top_k or 10,
            use_hybrid=True,
            fusion_method=effective_fusion,
            conversation_id=request.conversation_id,
            temperature=None,
            model_profile=effective_model_profile,
            use_llm=request.use_llm if request.use_llm is not None else True
        )

        processing_time = time.time() - start_time

        # Format retrieved documents
        retrieved_docs = []
        for doc in result.get("retrieved_documents", []):
            retrieved_docs.append(
                RetrievedDocument(
                    content=doc.get("text", ""),
                    score=doc.get("importance_score", doc.get("score", 0.0)),
                    metadata=doc.get("metadata", {}),
                    source_file=doc.get("metadata", {}).get("source_file", "unknown"),
                    chunk_index=doc.get("metadata", {}).get("chunk_index", 0)
                )
            )

        # Only include metadata if requested
        if not request.include_metadata:
            for doc in retrieved_docs:
                doc.metadata = {}

        return QueryResponse(
            query=request.query,
            answer=result.get("answer") if request.use_llm else None,
            retrieved_documents=retrieved_docs,
            retrieval_method=f"hybrid_{effective_fusion}",
            processing_time_seconds=processing_time,
            conversation_id=result.get("conversation_id"),
            model_used=result.get("model_used") if request.use_llm else None
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@app.get("/collections", tags=["Collections"])
async def list_collections():
    """
    List all available collections.

    Returns:
        List of collection names with stats
    """
    try:
        collections = await brain.hippocampus.qdrant_service.client.get_collections()

        collection_info = []
        for collection in collections.collections:
            try:
                info = await brain.hippocampus.get_memory_stats(collection.name)
                collection_info.append(info)
            except Exception as e:
                logger.warning(f"Error getting info for collection {collection.name}: {str(e)}")
                collection_info.append({
                    "name": collection.name,
                    "error": str(e)
                })

        return {
            "collections": collection_info,
            "total": len(collection_info)
        }

    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing collections: {str(e)}"
        )


@app.get("/models/profiles", tags=["Models"])
async def get_model_profiles():
    """
    Get available model profiles with their configurations.

    Returns:
        Dictionary of model profiles with specs
    """
    try:
        profiles = brain.llm_service.model_manager.list_available_profiles()
        current_model = brain.llm_service.get_current_model()
        current_profile = brain.llm_service.current_profile or settings.MODEL_PROFILE

        return {
            "profiles": profiles,
            "current_profile": current_profile,
            "current_model": current_model
        }

    except Exception as e:
        logger.error(f"Error getting model profiles: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting model profiles: {str(e)}"
        )


@app.get("/models/available", tags=["Models"])
async def get_available_models():
    """
    Get list of configured model profiles.

    Note: MLX models are loaded from Hugging Face cache and are not
    enumerated via an API. This endpoint returns the configured profiles.

    Returns:
        List of configured model profiles with their settings
    """
    try:
        profiles = brain.llm_service.model_manager.list_available_profiles()
        current_model = brain.llm_service.get_current_model()
        current_profile = brain.llm_service.current_profile or settings.MODEL_PROFILE

        model_list = []
        for profile_name, config in profiles.items():
            model_list.append({
                "profile": profile_name,
                "model": config.get("model", ""),
                "max_tokens": config.get("max_tokens", 0),
                "temperature": config.get("temperature", 0.0),
                "description": config.get("description", ""),
            })

        return {
            "models": model_list,
            "total": len(model_list),
            "current_model": current_model,
            "current_profile": current_profile,
            "note": "MLX models are cached from Hugging Face and loaded on demand"
        }

    except Exception as e:
        logger.error(f"Error listing available models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error listing available models: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG
    )
