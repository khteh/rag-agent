from deepagents.backends import CompositeBackend, StateBackend, StoreBackend, FilesystemBackend
# https://docs.langchain.com/oss/python/deepagents/backends
composite_backend = CompositeBackend(
    #default = StateBackend(rt), ephemeral
    default = FilesystemBackend(root_dir="output", virtual_mode=True),
    routes = {
        "/memories/": StoreBackend(), # Note that the tools access Store through the runtime.store
    }
)