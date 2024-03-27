import argparse
import bpy
import json
import mitsuba as mi
import numpy as np
import os
import pickle as pkl

from pygltflib import GLTF2

mi.set_variant("cuda_ad_rgb")


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path",
                        default="/home/zongtai/data/scenes/spheres.glb",
                        type=str, help="The path of the file to be converted")
    parser.add_argument("--to", type=str, help="The format to convert to")
    args = parser.parse_args()

    assert args.to in ["glb", "gltf", "xml"], "Invalid format to convert to"
    return args


def print_glb(scene):
    scene_dict = scene.__dict__
    names = [name for name in scene_dict if not name.startswith("_")]
    for name in names:
        content = scene_dict[name]
        if type(content) == list:
            print(f"{name}:")
            for i, item in enumerate(content):
                print(f"  {i}: {item}")
        else:
            print(f"{name}: {content}")


def load_buffer(accessor_ind, scene):
    accessor = scene.accessors[accessor_ind]
    bufferView = scene.bufferViews[accessor.bufferView]
    current_buffer = scene.buffers[bufferView.buffer]
    data = scene.get_data_from_buffer_uri(current_buffer.uri)

    nwtype = None
    sz = 3
    if accessor.type == "VEC3":
        nwtype = np.float32
        sz = 3
    elif accessor.type == "VEC2":
        nwtype = np.float32
        sz = 2
    elif accessor.type == "SCALAR":
        siz = bufferView.byteLength / accessor.count
        if siz == 2:
            nwtype = np.uint16
        elif siz == 4:
            nwtype = np.float32
        else:
            raise ValueError("Unsupported size")
        sz = 1

    ret = np.frombuffer(
        data[bufferView.byteOffset + accessor.byteOffset : bufferView.byteOffset + bufferView.byteLength],
        dtype=nwtype,
    ).reshape(-1, sz)
    return ret


def load_glb(scene):
    ret = {"type": "scene"}
    scene_list = scene.scenes
    for current_scene in scene_list:
        for node_idx in current_scene.nodes:
            node = scene.nodes[node_idx]
            mesh_idx = node.mesh
            current_mesh = {}
            current_mesh["type"] = "obj"
            current_mesh["filename"] = f"tmpfiles/{node.name}.obj"

            # read mesh
            mesh = scene.meshes[mesh_idx]
            assert len(mesh.primitives) == 1, "Only support single primitive mesh"
            for primitive in mesh.primitives:
                vertices = load_buffer(primitive.attributes.POSITION, scene)
                normals = load_buffer(primitive.attributes.NORMAL, scene)
                indices = load_buffer(primitive.indices, scene)
                triangles = []
                for i in range(0, indices.shape[0], 3):
                    triangles.append(indices[i : i + 3])
                triangles = np.array(triangles).squeeze()

                material = scene.materials[primitive.material]
                current_mesh["bsdf"] = {
                    "type": "principled",
                    "base_color": {
                        "type": "rgb",
                        "value": material.pbrMetallicRoughness.baseColorFactor[:3],
                    },
                    "metallic": material.pbrMetallicRoughness.metallicFactor,
                    "roughness": material.pbrMetallicRoughness.roughnessFactor,
                }

                # export to ply
                obj_path = current_mesh["filename"]
                with open(obj_path, "w") as f:
                    for v in vertices:
                        f.write(f"v {v[0]} {v[1]} {v[2]}\n")
                    for n in normals:
                        f.write(f"vn {n[0]} {n[1]} {n[2]}\n")
                    for t in triangles:
                        f.write(f"f {t[0]+1}//{t[0]+1} {t[1]+1}//{t[1]+1} {t[2]+1}//{t[2]+1}\n")

            # transformation
            transform = np.eye(4)
            if node.scale is not None:
                transform = np.dot(transform, np.diag(node.scale + [1]))
            if node.rotation is not None:
                q = node.rotation
                q = np.array(q)
                q = q / np.linalg.norm(q)
                x, y, z, w = q
                transform = np.dot(
                    transform,
                    np.array([[1 - 2*(y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w), 0],
                            [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w), 0],
                            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2), 0],
                            [0, 0, 0, 1]]),
                )
            if node.translation is not None:
                transform[:3, 3] = node.translation

            current_mesh["to_world"] = transform.tolist()
            ret[node.name] = current_mesh

    return ret


if __name__ == "__main__":
    args = load_args()
    file_name = args.file_path
    dir_path = "/".join(file_name.split("/")[:-1])
    os.makedirs("tmpfiles", exist_ok=True)

    if "." not in file_name:
        raise ValueError("File name must contain a file type")

    file_type = file_name.split(".")[-1]
    if file_type not in ["blend", "glb", "gltf", "xml"]:
        raise ValueError("File type not supported")

    if file_type == "blend":
        bpy.ops.wm.open_mainfile(filepath=file_name)
        bpy.ops.export_scene.gltf(
            filepath="tmpfiles/tmp.glt[f",
            export_format="GLB",
            export_animations=False,
        )
        file_name = "tmpfiles/tmp.glb"
        file_type = "glb"

    assert(file_type=="glb")
    assert(args.to=="xml")

    scene = GLTF2().load(file_name)
    scene = load_glb(scene)

    json.dump(scene, open("tmpfiles/scene.json", "w"), indent=4)
    with open("tmpfiles/scene.pkl", "wb") as f:
        pkl.dump(scene, f)

    with open("tmpfiles/scene.pkl", "rb") as f:
        scene = pkl.load(f)

    for k, v in scene.items():
        if type(v) == dict:
            for kk, vv in v.items():
                if kk == "to_world":
                    print(k)
                    print(vv)
                    scene[k][kk] = mi.ScalarTransform4f(vv)

    # render the scene
    scene["integrator"] = {
        "type": "path",
        "max_depth": 5,
    }
    # scene["light"] = {
    #     "type": "point",
    #     "position": [0, 15, -6],
    #     "intensity": {
    #         "type": "spectrum",
    #         "value": 1000.0,
    #     }
    # }
    scene["light2"] = {
        "type": "constant",
        "radiance": {
            "type": "rgb",
            "value": 0.4,
        }
    }
    scene = mi.load_dict(scene)

    # render
    sensor = mi.load_dict({
        # "type": "orthographic",
        "type": "perspective",
        "fov": 40,
        "fov_axis": "y",
        "to_world": mi.ScalarTransform4f.look_at(
            origin=[10, 10, -5],
            target=[0, 0, -1],
            up=[0, 0, -1],
        ),
        "film": {
            "type": "hdrfilm",
            "width": 1280,
            "height": 720,
        }
    })
    image = mi.render(scene, sensor=sensor, spp=256)
    mi.util.write_bitmap("tmpfiles/output.png", image)

    # uncomment after debug
    # os.system("rm -rf tmpfiles")