from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    rust_extensions=[
        RustExtension(
            "networkx_rust._rust",
            path="crates/nx-dijkstra-core/Cargo.toml",
            binding=Binding.PyO3,
        )
    ],
)
