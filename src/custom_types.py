from enum import Enum


class DeviceType(Enum):
    CUDA = "cuda"
    CPU = "cpu"


class GenerationType(Enum):
    RAG = "RAG"
    CAG = "CAG"