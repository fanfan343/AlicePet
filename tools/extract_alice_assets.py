#!/usr/bin/env python3
"""
Extract downsampled standalone props (.glb + TexLite PNGs) and Unity .anim recipes
(expressions, hands, poses/AFK, outfits/hair) from AlicePhil(VRC&Other) 1.05
into /Users/funnyliu/Downloads/VRM项目/降采样模型/独立道具与配方/.
"""

import json
import math
import os
import re
import struct
import subprocess
import tarfile
import zlib
from collections import defaultdict

SRC_PACK_DIR = "/Users/funnyliu/Downloads/AlicePhil(VRC&Other) 1.05"
UNITY_PKG = os.path.join(SRC_PACK_DIR, "VRCUnitypackage", "AlicePhilVRC.unitypackage")
OUT_DIR = "/Users/funnyliu/Downloads/VRM项目/降采样模型/独立道具与配方"
TEX_SRC_COLOR = os.path.join(SRC_PACK_DIR, "Textures", "PNG", "Basecolor_Item.png")
TEX_SRC_NORMAL = os.path.join(SRC_PACK_DIR, "Textures", "PNG", "Normal_Item.png")


# ---------------------------------------------------------------------------
# Minimal 3D matrix / quaternion helpers (column-major glTF & row-major FBX)
# ---------------------------------------------------------------------------

def mat4_identity():
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def mat4_mul(a, b):
    out = [[0.0] * 4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            out[i][j] = sum(a[i][k] * b[k][j] for k in range(4))
    return out


def mat4_vec3_point(m, v):
    x, y, z = v
    return (
        m[0][0] * x + m[0][1] * y + m[0][2] * z + m[0][3],
        m[1][0] * x + m[1][1] * y + m[1][2] * z + m[1][3],
        m[2][0] * x + m[2][1] * y + m[2][2] * z + m[2][3],
    )


def mat4_vec3_dir(m, v):
    x, y, z = v
    nx = m[0][0] * x + m[0][1] * y + m[0][2] * z
    ny = m[1][0] * x + m[1][1] * y + m[1][2] * z
    nz = m[2][0] * x + m[2][1] * y + m[2][2] * z
    ln = math.sqrt(nx * nx + ny * ny + nz * nz)
    if ln > 1e-12:
        return (nx / ln, ny / ln, nz / ln)
    return (0.0, 1.0, 0.0)


def mat4_inverse(m):
    # Gauss-Jordan 4x4 inverse
    a = [row[:] + [1.0 if i == j else 0.0 for j in range(4)] for i, row in enumerate(m)]
    for col in range(4):
        piv = max(range(col, 4), key=lambda r: abs(a[r][col]))
        a[col], a[piv] = a[piv], a[col]
        div = a[col][col]
        if abs(div) < 1e-12:
            return mat4_identity()
        for j in range(8):
            a[col][j] /= div
        for r in range(4):
            if r != col and abs(a[r][col]) > 1e-15:
                fac = a[r][col]
                for j in range(8):
                    a[r][j] -= fac * a[col][j]
    return [row[4:] for row in a]


def mat4_to_col_major(m):
    # glTF expects column-major 16 floats
    return [m[r][c] for c in range(4) for r in range(4)]


def euler_xyz_deg_to_quat(rx, ry, rz):
    # FBX intrinsic/extrinsic XYZ Euler: Rz * Ry * Rx in column-vector convention
    ax, ay, az = math.radians(rx) * 0.5, math.radians(ry) * 0.5, math.radians(rz) * 0.5
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)
    # q = qz * qy * qx
    qx = sx * cy * cz - cx * sy * sz
    qy = cx * sy * cz + sx * cy * sz
    qz = cx * cy * sz - sx * sy * cz
    qw = cx * cy * cz + sx * sy * sz
    ln = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    return [qx / ln, qy / ln, qz / ln, qw / ln]


def trs_to_mat4(t, q, s):
    x, y, z, w = q
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    sx, sy, sz = s
    return [
        [(1 - 2 * (yy + zz)) * sx, 2 * (xy - wz) * sy, 2 * (xz + wy) * sz, t[0]],
        [2 * (xy + wz) * sx, (1 - 2 * (xx + zz)) * sy, 2 * (yz - wx) * sz, t[1]],
        [2 * (xz - wy) * sx, 2 * (yz + wx) * sy, (1 - 2 * (xx + yy)) * sz, t[2]],
        [0.0, 0.0, 0.0, 1.0],
    ]


# ---------------------------------------------------------------------------
# Binary FBX 7400 parser -> glTF 2.0 (.glb)
# ---------------------------------------------------------------------------

def parse_fbx_file(path):
    with open(path, "rb") as f:
        hdr = f.read(27)
        if not hdr.startswith(b"Kaydara FBX Binary"):
            raise ValueError(f"Not binary FBX: {path}")
        ver = struct.unpack("<I", hdr[23:27])[0]
        is_64 = ver >= 7500

        def read_prop():
            t = f.read(1).decode("ascii")
            if t in ("Y", "C", "I", "F", "D", "L"):
                fmt = {"Y": "<h", "C": "<B", "I": "<i", "F": "<f", "D": "<d", "L": "<q"}[t]
                return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]
            elif t in ("S", "R"):
                l = struct.unpack("<I", f.read(4))[0]
                d = f.read(l)
                return d.decode("utf-8", errors="replace") if t == "S" else d
            elif t in ("f", "d", "l", "i", "b"):
                alen, enc, clen = struct.unpack("<III", f.read(12))
                raw = f.read(clen)
                if enc == 1:
                    raw = zlib.decompress(raw)
                fmt_ch = {"f": "f", "d": "d", "l": "q", "i": "i", "b": "B"}[t]
                return list(struct.unpack("<" + str(alen) + fmt_ch, raw))
            raise ValueError(f"Unknown FBX prop type {t}")

        def read_node():
            if is_64:
                end_offset, num_props, prop_len, nlen = struct.unpack("<QQQB", f.read(25))
            else:
                end_offset, num_props, prop_len, nlen = struct.unpack("<IIIB", f.read(13))
            if end_offset == 0:
                return None
            name = f.read(nlen).decode("utf-8", errors="replace")
            props = [read_prop() for _ in range(num_props)]
            children = []
            while f.tell() < end_offset:
                ch = read_node()
                if ch is None:
                    break
                children.append(ch)
            f.seek(end_offset)
            return {"name": name, "props": props, "children": children}

        roots = []
        while True:
            n = read_node()
            if n is None:
                break
            roots.append(n)
        return roots


def find_child(node, name):
    for c in node["children"]:
        if c["name"] == name:
            return c
    return None


def find_children(node, name):
    return [c for c in node["children"] if c["name"] == name]


def extract_fbx_scene_to_glb(fbx_path, glb_path, color_png_bytes, normal_png_bytes, bake_static_root_transform=False):
    roots = parse_fbx_file(fbx_path)
    objects_node = next(n for n in roots if n["name"] == "Objects")
    conns_node = next(n for n in roots if n["name"] == "Connections")

    models = {}
    geometries = {}
    materials = {}
    deformers = {}

    for obj in objects_node["children"]:
        nm = obj["name"]
        props = obj["props"]
        if not props:
            continue
        oid = props[0]
        if nm == "Model":
            label = props[1].split("\x00")[0]
            subtype = props[2]
            t = [0.0, 0.0, 0.0]
            r = [0.0, 0.0, 0.0]
            s = [1.0, 1.0, 1.0]
            p70 = find_child(obj, "Properties70")
            if p70:
                for p in find_children(p70, "P"):
                    pname = p["props"][0]
                    if pname == "Lcl Translation":
                        t = [float(x) for x in p["props"][4:7]]
                    elif pname == "Lcl Rotation":
                        r = [float(x) for x in p["props"][4:7]]
                    elif pname == "Lcl Scaling":
                        s = [float(x) for x in p["props"][4:7]]
            models[oid] = {"id": oid, "name": label, "subtype": subtype, "t": t, "r": r, "s": s}
        elif nm == "Geometry":
            label = props[1].split("\x00")[0]
            subtype = props[2]
            geometries[oid] = {"id": oid, "name": label, "subtype": subtype, "node": obj}
        elif nm == "Material":
            label = props[1].split("\x00")[0]
            materials[oid] = {"id": oid, "name": label}
        elif nm == "Deformer":
            label = props[1].split("\x00")[0]
            subtype = props[2]
            deformers[oid] = {"id": oid, "name": label, "subtype": subtype, "node": obj}

    # Build OO/OP connections
    children_of = defaultdict(list)
    parents_of = defaultdict(list)
    for c in conns_node["children"]:
        if c["name"] == "C" and len(c["props"]) >= 3:
            src, dst = c["props"][1], c["props"][2]
            children_of[dst].append(src)
            parents_of[src].append(dst)

    # Identify model hierarchy
    root_model_ids = [mid for mid in models if 0 in parents_of[mid]]
    model_children = defaultdict(list)
    for mid in models:
        for pid in parents_of[mid]:
            if pid in models:
                model_children[pid].append(mid)

    # Create glTF nodes for models in order
    model_id_to_gltf_idx = {}
    gltf_nodes = []
    ordered_model_ids = []

    def assign_model_idx(mid):
        if mid in model_id_to_gltf_idx:
            return
        model_id_to_gltf_idx[mid] = len(ordered_model_ids)
        ordered_model_ids.append(mid)
        for ch in model_children[mid]:
            assign_model_idx(ch)

    for rmid in root_model_ids:
        assign_model_idx(rmid)
    for mid in models:
        assign_model_idx(mid)

    for mid in ordered_model_ids:
        m = models[mid]
        q = euler_xyz_deg_to_quat(*m["r"])
        gnode = {
            "name": m["name"],
            "translation": m["t"],
            "rotation": q,
            "scale": m["s"],
        }
        ch_indices = [model_id_to_gltf_idx[c] for c in model_children[mid] if c in model_id_to_gltf_idx]
        if ch_indices:
            gnode["children"] = ch_indices
        gltf_nodes.append(gnode)

    # Compute world matrices of FBX model nodes (used for skinning / geometry check)
    world_mats = {}
    def compute_world(mid, parent_mat):
        m = models[mid]
        local_m = trs_to_mat4(m["t"], euler_xyz_deg_to_quat(*m["r"]), m["s"])
        wm = mat4_mul(parent_mat, local_m)
        world_mats[mid] = wm
        for ch in model_children[mid]:
            compute_world(ch, wm)

    for rmid in root_model_ids:
        compute_world(rmid, mat4_identity())

    # Binary glTF buffer builder
    bin_buf = bytearray()

    def append_bytes(data: bytes, align: int = 4):
        while len(bin_buf) % align != 0:
            bin_buf.append(0)
        off = len(bin_buf)
        bin_buf.extend(data)
        return off, len(data)

    buffer_views = []
    accessors = []

    def add_buffer_view(data: bytes, target=None):
        off, length = append_bytes(data, 4)
        bv = {"buffer": 0, "byteOffset": off, "byteLength": length}
        if target is not None:
            bv["target"] = target
        idx = len(buffer_views)
        buffer_views.append(bv)
        return idx

    def add_accessor(bv_idx, comp_type, count, acc_type, min_v=None, max_v=None):
        acc = {
            "bufferView": bv_idx,
            "componentType": comp_type,
            "count": count,
            "type": acc_type,
        }
        if min_v is not None:
            acc["min"] = min_v
        if max_v is not None:
            acc["max"] = max_v
        idx = len(accessors)
        accessors.append(acc)
        return idx

    # Embed textures (image 0 = basecolor, image 1 = normal)
    bv_col = add_buffer_view(color_png_bytes)
    bv_nrm = add_buffer_view(normal_png_bytes)
    images = [
        {"bufferView": bv_col, "mimeType": "image/png"},
        {"bufferView": bv_nrm, "mimeType": "image/png"},
    ]
    samplers = [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]
    textures = [{"sampler": 0, "source": 0}, {"sampler": 0, "source": 1}]

    # Two materials mapped from FBX materials: opaque ("Item") and transparent ("Item Transparent")
    gltf_materials = [
        {
            "name": "Item",
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0.0,
                "roughnessFactor": 0.88,
            },
            "normalTexture": {"index": 1, "scale": 0.65},
            "doubleSided": True,
        },
        {
            "name": "Item Transparent",
            "pbrMetallicRoughness": {
                "baseColorTexture": {"index": 0},
                "metallicFactor": 0.0,
                "roughnessFactor": 0.45,
                "baseColorFactor": [1.0, 1.0, 1.0, 0.85],
            },
            "normalTexture": {"index": 1, "scale": 0.5},
            "alphaMode": "BLEND",
            "doubleSided": True,
        },
    ]

    gltf_meshes = []
    gltf_skins = []

    for mid in ordered_model_ids:
        geom_ids = [cid for cid in children_of[mid] if cid in geometries and geometries[cid]["subtype"] == "Mesh"]
        if not geom_ids:
            continue
        gid = geom_ids[0]
        gnode = geometries[gid]["node"]

        verts_raw = find_child(gnode, "Vertices")["props"][0]
        poly_raw = find_child(gnode, "PolygonVertexIndex")["props"][0]
        num_ctrl_pts = len(verts_raw) // 3

        # LayerElementNormal
        norm_layer = find_child(gnode, "LayerElementNormal")
        norms_raw = find_child(norm_layer, "Normals")["props"][0] if norm_layer else None
        norm_map = find_child(norm_layer, "MappingInformationType")["props"][0] if norm_layer else ""
        norm_ref = find_child(norm_layer, "ReferenceInformationType")["props"][0] if norm_layer else ""
        norm_idx = find_child(norm_layer, "NormalsIndex")["props"][0] if (norm_layer and find_child(norm_layer, "NormalsIndex")) else None

        # LayerElementUV
        uv_layer = find_child(gnode, "LayerElementUV")
        uvs_raw = find_child(uv_layer, "UV")["props"][0] if uv_layer else None
        uv_map = find_child(uv_layer, "MappingInformationType")["props"][0] if uv_layer else ""
        uv_ref = find_child(uv_layer, "ReferenceInformationType")["props"][0] if uv_layer else ""
        uv_idx = find_child(uv_layer, "UVIndex")["props"][0] if (uv_layer and find_child(uv_layer, "UVIndex")) else None

        # LayerElementMaterial
        mat_layer = find_child(gnode, "LayerElementMaterial")
        mat_map = find_child(mat_layer, "MappingInformationType")["props"][0] if mat_layer else "AllSame"
        mat_indices = find_child(mat_layer, "Materials")["props"][0] if (mat_layer and find_child(mat_layer, "Materials")) else [0]

        # Connected FBX Materials on this Model
        model_mats = [cid for cid in children_of[mid] if cid in materials]
        def is_transparent_fbx_mat(local_mat_slot):
            if local_mat_slot < len(model_mats):
                mname = materials[model_mats[local_mat_slot]]["name"].lower()
                return 1 if "transparent" in mname else 0
            return 0

        # Skinning info (Cluster deformers) - skip when baking static props cleanly into world space
        skin_deformers = [] if bake_static_root_transform else [cid for cid in children_of[gid] if cid in deformers and deformers[cid]["subtype"] == "Skin"]
        ctrl_joints = [[0, 0, 0, 0] for _ in range(num_ctrl_pts)]
        ctrl_weights = [[1.0, 0.0, 0.0, 0.0] for _ in range(num_ctrl_pts)]
        skin_idx = None
        bake_world = world_mats.get(mid, mat4_identity()) if bake_static_root_transform else None

        if skin_deformers:
            sdef_id = skin_deformers[0]
            cluster_ids = [cid for cid in children_of[sdef_id] if cid in deformers and deformers[cid]["subtype"] == "Cluster"]
            joint_node_indices = []
            ibm_floats = []
            # Accumulate up to 4 weights per control point
            cp_influences = [[] for _ in range(num_ctrl_pts)]

            mesh_world = world_mats.get(mid, mat4_identity())

            for cl_id in cluster_ids:
                cl_node = deformers[cl_id]["node"]
                bone_ids = [cid for cid in children_of[cl_id] if cid in models]
                if not bone_ids:
                    continue
                bone_id = bone_ids[0]
                local_joint_idx = len(joint_node_indices)
                joint_node_indices.append(model_id_to_gltf_idx[bone_id])

                # Compute inverse bind matrix in mesh local space:
                # Vertex in FBX Geometry space is transformed by mesh_world in bind pose;
                # GLTF inverseBindMatrix expects world_of_bone^-1 * mesh_world!
                bone_world = world_mats.get(bone_id, mat4_identity())
                # If FBX TransformLink exists, check: FBX TransformLink is bone global bind matrix;
                # Transform is mesh global bind matrix. Both match world_mats!
                tl_node = find_child(cl_node, "TransformLink")
                tr_node = find_child(cl_node, "Transform")
                if tl_node and tr_node:
                    tl = tl_node["props"][0]
                    tr = tr_node["props"][0]
                    # FBX 4x4 arrays are column-major row-vectors (i.e. transpose of column-major matrix = row-major transpose)
                    # Actually in FBX, matrix stored as 16 floats where elements 12,13,14 are translation!
                    # So M[row][col] = arr[col * 4 + row]
                    bone_bind = [[tl[c * 4 + r] for c in range(4)] for r in range(4)]
                    mesh_bind = [[tr[c * 4 + r] for c in range(4)] for r in range(4)]
                    ibm = mat4_mul(mat4_inverse(bone_bind), mesh_bind)
                else:
                    ibm = mat4_mul(mat4_inverse(bone_world), mesh_world)

                ibm_floats.extend(mat4_to_col_major(ibm))

                idx_node = find_child(cl_node, "Indexes")
                wgt_node = find_child(cl_node, "Weights")
                if idx_node and wgt_node:
                    for cp_i, w_val in zip(idx_node["props"][0], wgt_node["props"][0]):
                        if 0 <= cp_i < num_ctrl_pts and w_val > 1e-6:
                            cp_influences[cp_i].append((local_joint_idx, float(w_val)))

            for cp_i, infls in enumerate(cp_influences):
                if infls:
                    infls.sort(key=lambda x: -x[1])
                    top4 = infls[:4]
                    wsum = sum(w for _, w in top4)
                    j4 = [j for j, _ in top4] + [0] * (4 - len(top4))
                    w4 = [w / wsum for _, w in top4] + [0.0] * (4 - len(top4))
                    ctrl_joints[cp_i] = j4
                    ctrl_weights[cp_i] = w4

            if joint_node_indices:
                ibm_bytes = struct.pack("<" + "f" * len(ibm_floats), *ibm_floats)
                bv_ibm = add_buffer_view(ibm_bytes)
                acc_ibm = add_accessor(bv_ibm, 5126, len(joint_node_indices), "MAT4")
                skin_idx = len(gltf_skins)
                gltf_skins.append({
                    "inverseBindMatrices": acc_ibm,
                    "joints": joint_node_indices,
                })

        # Unpack polygons and triangulate with fan, grouped by gltf material index (0 or 1)
        prims_by_mat = defaultdict(list)  # mat_idx -> list of (cp_idx, norm, uv)
        poly_start = 0
        poly_num = 0
        face_verts = []
        for pv_idx, raw_v in enumerate(poly_raw):
            is_end = raw_v < 0
            cp_idx = (~raw_v) if is_end else raw_v

            # Normal at polygon vertex pv_idx (or control point)
            if norms_raw:
                if norm_map == "ByPolygonVertex":
                    ni = norm_idx[pv_idx] if (norm_ref == "IndexToDirect" and norm_idx) else pv_idx
                elif norm_map == "ByVertice":
                    ni = norm_idx[cp_idx] if (norm_ref == "IndexToDirect" and norm_idx) else cp_idx
                else:
                    ni = 0
                nx, ny, nz = norms_raw[ni * 3], norms_raw[ni * 3 + 1], norms_raw[ni * 3 + 2]
            else:
                nx, ny, nz = 0.0, 1.0, 0.0

            # UV at polygon vertex pv_idx (flip V for glTF top-left image origin!)
            if uvs_raw:
                if uv_map == "ByPolygonVertex":
                    ui = uv_idx[pv_idx] if (uv_ref == "IndexToDirect" and uv_idx) else pv_idx
                elif uv_map == "ByVertice":
                    ui = uv_idx[cp_idx] if (uv_ref == "IndexToDirect" and uv_idx) else cp_idx
                else:
                    ui = 0
                u, v = uvs_raw[ui * 2], 1.0 - uvs_raw[ui * 2 + 1]
            else:
                u, v = 0.0, 0.0

            face_verts.append((cp_idx, (float(nx), float(ny), float(nz)), (float(u), float(v))))

            if is_end:
                if mat_map == "ByPolygon" and poly_num < len(mat_indices):
                    gmat = is_transparent_fbx_mat(mat_indices[poly_num])
                else:
                    gmat = is_transparent_fbx_mat(mat_indices[0] if mat_indices else 0)
                for k in range(1, len(face_verts) - 1):
                    prims_by_mat[gmat].extend([face_verts[0], face_verts[k], face_verts[k + 1]])
                face_verts = []
                poly_num += 1

        gltf_primitives = []
        for gmat, tri_verts in sorted(prims_by_mat.items()):
            pos_list = []
            nrm_list = []
            uv_list = []
            jnt_list = []
            wgt_list = []
            idx_list = []
            vert_dedup = {}

            pmin = [1e9, 1e9, 1e9]
            pmax = [-1e9, -1e9, -1e9]

            for cp_idx, nrm, uv in tri_verts:
                key = (cp_idx, round(nrm[0], 5), round(nrm[1], 5), round(nrm[2], 5), round(uv[0], 5), round(uv[1], 5))
                if key not in vert_dedup:
                    v_idx = len(pos_list) // 3
                    vert_dedup[key] = v_idx
                    px, py, pz = verts_raw[cp_idx * 3], verts_raw[cp_idx * 3 + 1], verts_raw[cp_idx * 3 + 2]
                    if bake_world is not None:
                        px, py, pz = mat4_vec3_point(bake_world, (px, py, pz))
                        bnx, bny, bnz = mat4_vec3_dir(bake_world, nrm)
                        nrm = (bnx, bny, bnz)
                    pos_list.extend([float(px), float(py), float(pz)])
                    pmin = [min(pmin[0], px), min(pmin[1], py), min(pmin[2], pz)]
                    pmax = [max(pmax[0], px), max(pmax[1], py), max(pmax[2], pz)]
                    nrm_list.extend(nrm)
                    uv_list.extend(uv)
                    if skin_idx is not None:
                        jnt_list.extend(ctrl_joints[cp_idx])
                        wgt_list.extend(ctrl_weights[cp_idx])
                idx_list.append(vert_dedup[key])

            bv_pos = add_buffer_view(struct.pack("<" + "f" * len(pos_list), *pos_list), 34962)
            bv_nrm_a = add_buffer_view(struct.pack("<" + "f" * len(nrm_list), *nrm_list), 34962)
            bv_uv = add_buffer_view(struct.pack("<" + "f" * len(uv_list), *uv_list), 34962)
            bv_idx = add_buffer_view(struct.pack("<" + "H" * len(idx_list), *idx_list), 34963)

            vcount = len(pos_list) // 3
            attrs = {
                "POSITION": add_accessor(bv_pos, 5126, vcount, "VEC3", pmin, pmax),
                "NORMAL": add_accessor(bv_nrm_a, 5126, vcount, "VEC3"),
                "TEXCOORD_0": add_accessor(bv_uv, 5126, vcount, "VEC2"),
            }
            if skin_idx is not None:
                bv_jnt = add_buffer_view(struct.pack("<" + "H" * len(jnt_list), *jnt_list), 34962)
                bv_wgt = add_buffer_view(struct.pack("<" + "f" * len(wgt_list), *wgt_list), 34962)
                attrs["JOINTS_0"] = add_accessor(bv_jnt, 5123, vcount, "VEC4")
                attrs["WEIGHTS_0"] = add_accessor(bv_wgt, 5126, vcount, "VEC4")

            gltf_primitives.append({
                "attributes": attrs,
                "indices": add_accessor(bv_idx, 5123, len(idx_list), "SCALAR"),
                "material": gmat,
            })

        mesh_idx = len(gltf_meshes)
        gltf_meshes.append({"name": models[mid]["name"], "primitives": gltf_primitives})
        gnode_idx = model_id_to_gltf_idx[mid]
        gltf_nodes[gnode_idx]["mesh"] = mesh_idx
        if bake_world is not None:
            gltf_nodes[gnode_idx]["translation"] = [0.0, 0.0, 0.0]
            gltf_nodes[gnode_idx]["rotation"] = [0.0, 0.0, 0.0, 1.0]
            gltf_nodes[gnode_idx]["scale"] = [1.0, 1.0, 1.0]
        if skin_idx is not None:
            gltf_nodes[gnode_idx]["skin"] = skin_idx

    while len(bin_buf) % 4 != 0:
        bin_buf.append(0)

    scene_nodes = [model_id_to_gltf_idx[rmid] for rmid in root_model_ids]
    gltf_animations = []

    # Optional: if this is Alice_Items, synthesize a smooth 8-second looping TeaTimeOrbit animation
    if os.path.basename(fbx_path) == "Alice_Items.fbx":
        tea_bones = [
            "CupPlate", "Teapot", "Cake", "Puff_Cake", "Clock",
            "Cookie1", "Cookie2", "Cookie3", "Pocker1", "Pocker2", "Key"
        ]
        name_to_mid = {models[mid]["name"]: mid for mid in ordered_model_ids}
        root_mid = name_to_mid.get("Root")
        if root_mid is not None:
            inv_root_world = mat4_inverse(world_mats[root_mid])
            duration = 8.0
            steps = 33
            times = [round(duration * i / (steps - 1), 5) for i in range(steps)]
            bv_time = add_buffer_view(struct.pack("<" + "f" * len(times), *times))
            acc_time = add_accessor(bv_time, 5126, len(times), "SCALAR", [times[0]], [times[-1]])

            samplers = []
            channels = []
            # World-space front/side semicircle around Alice so all 11 tea props stay visible
            for idx, bname in enumerate(tea_bones):
                bmid = name_to_mid.get(bname)
                if bmid is None:
                    continue
                gnode_idx = model_id_to_gltf_idx[bmid]
                # Spread across left (-0.75..-0.28) and right (+0.28..+0.75) around Alice
                side = -1.0 if idx % 2 == 0 else 1.0
                ring_pos = idx // 2
                radius_x = side * (0.34 + 0.07 * (ring_pos % 3))
                base_y = 0.56 + 0.14 * ring_pos
                base_z = 0.16 + 0.08 * ((idx % 3) - 1)

                trans_vals = []
                rot_vals = []
                for step_i in range(steps):
                    phase = (2.0 * math.pi * step_i) / (steps - 1)
                    wx = radius_x + 0.03 * math.sin(phase + idx)
                    wz = base_z + 0.04 * math.cos(phase + idx)
                    wy = base_y + 0.05 * math.sin(phase * 2.0 + idx)
                    lx, ly, lz = mat4_vec3_point(inv_root_world, (wx, wy, wz))
                    trans_vals.extend([lx, ly, lz])

                    # Upright + gentle self-spin
                    spin_deg = (360.0 * step_i) / (steps - 1) + idx * 32.0
                    q = euler_xyz_deg_to_quat(180.0, spin_deg, 0.0)
                    rot_vals.extend(q)

                bv_t = add_buffer_view(struct.pack("<" + "f" * len(trans_vals), *trans_vals))
                acc_t = add_accessor(bv_t, 5126, len(times), "VEC3")
                s_t_idx = len(samplers)
                samplers.append({"input": acc_time, "output": acc_t, "interpolation": "LINEAR"})
                channels.append({"sampler": s_t_idx, "target": {"node": gnode_idx, "path": "translation"}})

                bv_r = add_buffer_view(struct.pack("<" + "f" * len(rot_vals), *rot_vals))
                acc_r = add_accessor(bv_r, 5126, len(times), "VEC4")
                s_r_idx = len(samplers)
                samplers.append({"input": acc_time, "output": acc_r, "interpolation": "LINEAR"})
                channels.append({"sampler": s_r_idx, "target": {"node": gnode_idx, "path": "rotation"}})

            if channels:
                gltf_animations.append({"name": "TeaTimeOrbit", "samplers": samplers, "channels": channels})

    gltf_doc = {
        "asset": {"version": "2.0", "generator": "AlicePet-FBX2GLB"},
        "scene": 0,
        "scenes": [{"nodes": scene_nodes}],
        "nodes": gltf_nodes,
        "meshes": gltf_meshes,
        "materials": gltf_materials,
        "textures": textures,
        "images": images,
        "samplers": samplers,
        "bufferViews": buffer_views,
        "accessors": accessors,
        "buffers": [{"byteLength": len(bin_buf)}],
    }
    if gltf_animations:
        gltf_doc["animations"] = gltf_animations
    if gltf_skins:
        gltf_doc["skins"] = gltf_skins

    json_bytes = json.dumps(gltf_doc, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    while len(json_bytes) % 4 != 0:
        json_bytes += b" "

    total_len = 12 + 8 + len(json_bytes) + 8 + len(bin_buf)
    with open(glb_path, "wb") as out:
        out.write(struct.pack("<III", 0x46546C67, 2, total_len))
        out.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))
        out.write(json_bytes)
        out.write(struct.pack("<II", len(bin_buf), 0x004E4942))
        out.write(bytes(bin_buf))

    print(f"Wrote {glb_path}: {total_len/1024:.1f} KB, meshes={len(gltf_meshes)}, skins={len(gltf_skins)}, nodes={len(gltf_nodes)}")


# ---------------------------------------------------------------------------
# Unity .anim YAML extractor -> recipes.json
# ---------------------------------------------------------------------------

def decode_unity_attr(raw: str) -> str:
    s = raw.strip()
    if s.startswith('"') and s.endswith('"'):
        s = s[1:-1]
    # Decode \uXXXX escapes
    s = re.sub(r"\\u([0-9a-fA-F]{4})", lambda m: chr(int(m.group(1), 16)), s)
    return s


def parse_anim_t0(txt: str):
    """Extract float/vec3 attributes at first keyframe across Unity serializedVersion 6 and 7."""
    float_curves = {}
    pos_curves = {}
    euler_curves = {}

    def slice_section(section_name):
        idx = txt.find(section_name)
        if idx == -1:
            return ""
        rest = txt[idx + len(section_name):]
        end_m = re.search(r"^\s{2}m_[A-Za-z0-9_]+:", rest, re.M)
        return rest[:end_m.start()] if end_m else rest

    # Match each curve block containing m_Curve -> first keyframe -> attribute -> path
    float_chunk = slice_section("m_FloatCurves:")
    curve_pattern = re.compile(
        r"m_Curve:\s*\n(?:\s*-\s*serializedVersion:\s*\d+\s*\n\s*time:\s*[-0-9.eE+]+\s*\n\s*value:\s*([^\n]+).*?)"
        r"attribute:\s*([^\n]+)\s*\n\s*path:\s*([^\n]*)",
        re.S,
    )
    for m in curve_pattern.finditer(float_chunk):
        vraw, attr_raw, path_raw = m.group(1).strip(), m.group(2), m.group(3).strip()
        attr = decode_unity_attr(attr_raw)
        try:
            float_curves[(path_raw, attr)] = float(vraw)
        except ValueError:
            pass

    vec3_pattern = re.compile(
        r"m_Curve:\s*\n(?:\s*-\s*serializedVersion:\s*\d+\s*\n\s*time:\s*[-0-9.eE+]+\s*\n\s*value:\s*\{x:\s*([-0-9.eE+]+),\s*y:\s*([-0-9.eE+]+),\s*z:\s*([-0-9.eE+]+)\}.*?)"
        r"path:\s*([^\n]*)",
        re.S,
    )
    for sec, store in [("m_PositionCurves:", pos_curves), ("m_EulerCurves:", euler_curves)]:
        chunk = slice_section(sec)
        for m in vec3_pattern.finditer(chunk):
            store[m.group(4).strip()] = [float(m.group(1)), float(m.group(2)), float(m.group(3))]

    return float_curves, pos_curves, euler_curves


def extract_recipes():
    guid_to_path = {}
    anim_texts = {}
    with tarfile.open(UNITY_PKG, "r:gz") as tf:
        members = tf.getmembers()
        for m in members:
            parts = m.name.split("/")
            if len(parts) == 2 and parts[1] == "pathname":
                guid_to_path[parts[0]] = tf.extractfile(m).read().decode("utf-8", errors="replace").splitlines()[0]
        for m in members:
            parts = m.name.split("/")
            if len(parts) == 2 and parts[1] == "asset":
                p = guid_to_path.get(parts[0], "")
                if p.endswith(".anim"):
                    anim_texts[p] = tf.extractfile(m).read().decode("utf-8", errors="replace")

    expressions = []
    hands = []
    outfits = {}
    poses = {}

    for path in sorted(anim_texts):
        txt = anim_texts[path]
        fname = os.path.basename(path).replace(".anim", "")
        fcurves, pcurves, ecurves = parse_anim_t0(txt)

        if "/Expresstions Face/" in path:
            if fname.startswith("##"):
                continue
            morphs = {}
            for (_, attr), val in fcurves.items():
                if attr.startswith("blendShape.") and abs(val) > 0.01:
                    bs_name = attr[len("blendShape."):]
                    morphs[bs_name] = round(val / 100.0, 4)
            if morphs:
                expressions.append({
                    "id": fname,
                    "label": fname,
                    "morphs": morphs,
                })

        elif "/Hands/" in path:
            left_muscles = {}
            right_muscles = {}
            for (_, attr), val in fcurves.items():
                if attr.startswith("LeftHand."):
                    left_muscles[attr[len("LeftHand."):]] = round(val, 4)
                elif attr.startswith("RightHand."):
                    right_muscles[attr[len("RightHand."):]] = round(val, 4)
            short = fname.replace("Alice_hands_", "")
            hands.append({
                "id": short,
                "label": short,
                "left": left_muscles,
                "right": right_muscles,
            })

        elif "/Part/" in path:
            morphs = {}
            node_toggles = {}
            for (obj_path, attr), val in fcurves.items():
                if attr.startswith("blendShape."):
                    morphs[attr[len("blendShape."):]] = round(val / 100.0, 4)
                elif attr == "m_IsActive" and obj_path:
                    node_toggles[obj_path] = (val > 0.5)
            bone_eulers = {}
            for obj_path, euler in ecurves.items():
                if obj_path:
                    bone_eulers[obj_path.split("/")[-1]] = euler
            outfits[fname] = {
                "morphs": morphs,
                "nodes": node_toggles,
                "boneEulers": bone_eulers,
            }

        elif "/Pose/" in path and fname in ("Alice_Stand_still", "Alice_Sit", "Alice_Crouch_still", "Alice_AFK_Loop"):
            muscles = {}
            prop_pos = {}
            prop_euler = {}
            for (obj_path, attr), val in fcurves.items():
                if not attr.startswith("blendShape.") and not attr.startswith("LeftHand.") and not attr.startswith("RightHand.") and attr != "m_IsActive":
                    if not attr.endswith("T.x") and not attr.endswith("T.y") and not attr.endswith("T.z") and not attr.endswith("Q.x") and not attr.endswith("Q.y") and not attr.endswith("Q.z") and not attr.endswith("Q.w"):
                        muscles[attr] = round(val, 4)
            for obj_path, vec in pcurves.items():
                if "Item" in obj_path or "Book" in obj_path:
                    prop_pos[obj_path] = vec
            for obj_path, vec in ecurves.items():
                if "Item" in obj_path or "Book" in obj_path:
                    prop_euler[obj_path] = vec
            poses[fname] = {
                "muscles": muscles,
                "propPositions": prop_pos,
                "propEulers": prop_euler,
            }

    return {
        "source": "AlicePhilVRC.unitypackage",
        "expressions": expressions,
        "hands": hands,
        "outfits": outfits,
        "poses": poses,
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Downsample Item textures to 1024px TexLite
    col_out = os.path.join(OUT_DIR, "Basecolor_Item_TexLite.png")
    nrm_out = os.path.join(OUT_DIR, "Normal_Item_TexLite.png")
    subprocess.run(["sips", "-Z", "1024", TEX_SRC_COLOR, "--out", col_out], check=True, stdout=subprocess.DEVNULL)
    subprocess.run(["sips", "-Z", "1024", TEX_SRC_NORMAL, "--out", nrm_out], check=True, stdout=subprocess.DEVNULL)

    with open(col_out, "rb") as f:
        col_bytes = f.read()
    with open(nrm_out, "rb") as f:
        nrm_bytes = f.read()

    # 2. Convert Alice_Item_Book.fbx (baked static) & Alice_Items.fbx (skinned + animated) -> .glb
    extract_fbx_scene_to_glb(
        os.path.join(SRC_PACK_DIR, "Alice_Item_Book.fbx"),
        os.path.join(OUT_DIR, "Alice_Item_Book.glb"),
        col_bytes,
        nrm_bytes,
        bake_static_root_transform=True,
    )
    extract_fbx_scene_to_glb(
        os.path.join(SRC_PACK_DIR, "Alice_Items.fbx"),
        os.path.join(OUT_DIR, "Alice_Items.glb"),
        col_bytes,
        nrm_bytes,
        bake_static_root_transform=False,
    )

    # 3. Extract recipes.json
    recipes = extract_recipes()
    recipes_path = os.path.join(OUT_DIR, "recipes.json")
    with open(recipes_path, "w", encoding="utf-8") as f:
        json.dump(recipes, f, indent=2, ensure_ascii=False)
    print(f"Wrote {recipes_path}: {len(recipes['expressions'])} expressions, {len(recipes['hands'])} hands, {len(recipes['outfits'])} outfits, {len(recipes['poses'])} poses")


if __name__ == "__main__":
    main()
