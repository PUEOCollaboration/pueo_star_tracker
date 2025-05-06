# Copyright (c) 2023 Steven Rosenthal smr@dt3.org
# See LICENSE file in root directory for license terms.

"""
This example loads the tetra3 default database and solves an image using CedarDetect's
centroid finding and Tetra3's solve_from_centroids().

Note: Requires PIL (pip install Pillow)
"""

import sys
import os
import json
import random
import time
sys.path.append('../..')

import pandas as pd

import numpy as np
from lib.cedar_solve import Tetra3
from PIL import Image
from pathlib import Path
from time import perf_counter as precision_timestamp

import grpc
from multiprocessing import shared_memory
import lib.cedar_detect.python.cedar_detect_pb2 as cedar_detect_pb2
import lib.cedar_detect.python.cedar_detect_pb2_grpc as cedar_detect_pb2_grpc


results_path = 'results'

def save_json(file: str, data):
    """Save to json file, handling non-serializable objects."""
    if not file.endswith('.json'):
        file += '.json'

    # Convert non-serializable objects in the data
    def convert_to_serializable(obj):
        if isinstance(obj, (pd.Series, np.ndarray)):
            return obj.tolist()  # Convert Series/NumPy array to list
        elif isinstance(obj, (np.int64, np.float64)):
            return int(obj) if isinstance(obj, np.int64) else float(obj)  # Convert NumPy types to Python types
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in
                    obj.items()}  # Recursively handle dictionaries
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]  # Recursively handle lists/tuples
        else:
            return obj  # Return as-is if already serializable

    # Convert the entire data structure
    serializable_data = convert_to_serializable(data)

    # Save to JSON file
    filename = os.path.join(results_path, file)
    with open(filename, 'w', encoding='utf-8') as json_file:
        json.dump(serializable_data, json_file, indent=4)  # indent=4 for pretty-printing

def extract_centroids(stub, image):
    cr = cedar_detect_pb2.CentroidsRequest(
        input_image=image,
        sigma=2.2,
        max_size=50,
        binning=False,
        return_binned=False,
        use_binned_for_star_candidates=False,
        detect_hot_pixels=False
    )
    return stub.ExtractCentroids(cr)

# Create instance and load default_database.
t3 = Tetra3('/home/milc/Projects/pcc/data/fov_2.5_mag10.0_tyc_v9_cedar_database.npz')  # 'database_auto_30_10_002')

# Set up to make gRPC calls to CedarDetect centroid finder (it must be running
# already).
channel = grpc.insecure_channel('localhost:50051')
stub = cedar_detect_pb2_grpc.CedarDetectStub(channel)

# Use shared memory to make the gRPC calls faster. This works only when the
# client (this program) and the CedarDetect gRPC server are running on the same
# machine.
USE_SHMEM = False

# Path where test images are.
path = Path('../test_data/')
idx = 0
images = list(path.glob('*.jpg')) + list(path.glob('*.bmp')) + list(path.glob('*.png'))
random.shuffle(images)
for impath in images:
    idx += 1
    t0 = time.monotonic()
    print('Solving for image at: ' + str(impath))
    with Image.open(str(impath)) as img:
        img = img.copy().convert(mode='L')
        (width, height) = (img.width, img.height)
        image = np.asarray(img, dtype=np.uint8)

        centroids_result = None
        rpc_duration_secs = None
        if USE_SHMEM:
            # Using shared memory. The image data is passed in a shared memory
            # object, with the gRPC request giving the name of the shared memory
            # object.

            # Set up shared memory object for passing input image to CedarDetect.
            name = f"/cedar_detect_image__{uuid.uuid4()}"
            shmem = shared_memory.SharedMemory(name, create=True, size=height*width)
            try:
                # Create numpy array backed by shmem.
                shimg = np.ndarray(image.shape, dtype=image.dtype, buffer=shmem.buf)
                # Copy image into shimg. This is much cheaper than passing image
                # over the gRPC call.
                shimg[:] = image[:]

                im = cedar_detect_pb2.Image(width=width, height=height, shmem_name=shmem.name)
                rpc_start = precision_timestamp()
                centroids_result = extract_centroids(stub, im)
                rpc_duration_secs = precision_timestamp() - rpc_start
            finally:
                shmem.close()
                shmem.unlink()
        else:
            # Not using shared memory. The image data is passed as part of the
            # gRPC request.
            im = cedar_detect_pb2.Image(width=width, height=height,
                                        image_data=image.tobytes())
            rpc_start = precision_timestamp()
            centroids_result = extract_centroids(stub, im)
            rpc_duration_secs = precision_timestamp() - rpc_start

        if len(centroids_result.star_candidates) == 0:
            print('Found no stars!')
        else:
            tetra_centroids = []  # List of (y, x).
            for sc in centroids_result.star_candidates:
                tetra_centroids.append((sc.centroid_position.y,
                                        sc.centroid_position.x))

            # Save Centroids
            save_json('tetra_centroids', tetra_centroids)
            size = (height, width)
            print(f'Solving for image size: {size}')
            solved = t3.solve_from_centroids(tetra_centroids,
                                             size=size,
                                             fov_estimate=2.5,
                                             return_matches=True
                                             )
            algo_duration_secs = (centroids_result.algorithm_time.seconds +
                                  centroids_result.algorithm_time.nanos / 1e9)
            # print(f'Centroids: {tetra_centroids}')
            print(f'  Solved: {time.monotonic() - t0:.4f}s. Centroids: %s. Solution: %s. %.2fms in centroiding (%.2fms rpc overhead)' %
                  (len(tetra_centroids),
                   solved,
                   rpc_duration_secs * 1000,
                   (rpc_duration_secs - algo_duration_secs) * 1000))
