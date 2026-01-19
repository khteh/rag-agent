from deepagents.backends import CompositeBackend, StateBackend, StoreBackend, FilesystemBackend
# https://docs.langchain.com/oss/python/deepagents/backends
composite_backend = lambda rt: CompositeBackend(
    #default = StateBackend(rt), ephemeral
    default = FilesystemBackend(root_dir="output", virtual_mode=True),
    routes = {
        "/memories/": StoreBackend(rt), # Note that the tools access Store through the runtime.store
    }
)