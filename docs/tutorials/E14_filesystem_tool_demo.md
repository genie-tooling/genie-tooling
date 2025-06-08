# Tutorial: FileSystem Tool (E14)

This tutorial corresponds to the example file `examples/E14_filesystem_tool_demo.py`.

It demonstrates how to configure and use the `SandboxedFileSystemTool` for safe file operations. It shows how to:
- Enable the tool and configure its `sandbox_base_path` in `tool_configurations`.
- Perform `write_file`, `read_file`, and `list_directory` operations.
- Verify that path traversal attempts outside the sandbox are prevented.

## Example Code

--8<-- "examples/E14_filesystem_tool_demo.py"
