from deepagents.backends import CompositeBackend, StateBackend, StoreBackend, FilesystemBackend
composite_backend = lambda rt: CompositeBackend(
    #default = StateBackend(rt), ephemeral
    default = FilesystemBackend(root_dir="output", virtual_mode=True),
    routes = {
        "/memories/": StoreBackend(rt),
    }
)