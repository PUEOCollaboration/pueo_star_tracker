# 2. With python/ as your current directory, execute:

../../../.venv/bin/python -m grpc_tools.protoc -I../src/proto --python_out=. --pyi_out=. --grpc_python_out=. ../src/proto/cedar_detect.proto


# 3. Run the CedarDetect gRPC server in background:

cargo run --release --bin cedar-detect-server
