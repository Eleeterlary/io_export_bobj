# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

# <pep8 compliant>

import bpy
import bmesh
import os
import shutil
import math
import mathutils
from bpy_extras.io_utils import ImportHelper, ExportHelper
from bpy.props import StringProperty, BoolProperty, FloatProperty
from bpy.types import Operator

bl_info = {
    "name": "Blockbuster OBJ Toolset (.bobj)",
    "author": "McHorse",
    "version": (1, 2, 0),
    "blender": (4, 1, 0),
    "location": "File > Import-Export",
    "description": "Import and Export BOBJ files",
    "warning": "",
    "wiki_url": "",
    "category": "Import-Export",
}



def name_compat(name):
    return 'None' if name is None else name.replace(' ', '_')

def mesh_triangulate(mesh):
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bmesh.ops.triangulate(bm, faces=bm.faces)
    bm.to_mesh(mesh)
    bm.free()

def stringify_keyframe(context, keyframe):
    fps = context.scene.render.fps
    f = 20 / fps

    interp = keyframe.interpolation
    result = 'kf %d %f %s' % (keyframe.co[0] * f, keyframe.co[1], interp)
    result += ' %f %f %f %f' % (keyframe.handle_left[0] * f, keyframe.handle_left[1], keyframe.handle_right[0] * f, keyframe.handle_right[1])

    return result


class ExportBOBJ(Operator, ExportHelper):
    """Export Blockbuster OBJ"""
    bl_idname = "export_scene.bobj"
    bl_label = "Export BOBJ"
    filename_ext = ".bobj"

    filter_glob: StringProperty(
        default="*.bobj",
        options={'HIDDEN'},
        maxlen=255,
    )

    use_selection: BoolProperty(
        name="Selection Only",
        description="Export selected objects only",
        default=True,
    )

    include_keyframes: BoolProperty(
        name="Include Animation",
        description="Export keyframe animation data",
        default=True,
    )

    export_convert_to_euler: BoolProperty(
        name="Convert Quaternions to XYZ",
        description="Convert Quaternion rotations to Euler XYZ during export",
        default=True,
    )

    export_textures: BoolProperty(
        name="Export Textures",
        description="If enabled, export textures associated with materials",
        default=False,
    )

    def execute(self, context):
        return save(context, self.filepath,
                    use_selection=self.use_selection,
                    include_keyframes=self.include_keyframes,
                    convert_to_euler=self.export_convert_to_euler,
                    export_textures=self.export_textures)

def write_armature(fw, armature, global_matrix):
    fw('arm_name %s\n' % armature.data.name)

    if armature.animation_data is not None and armature.animation_data.action is not None:
        fw ('arm_action %s\n' % name_compat(armature.animation_data.action.name))

    for bone in armature.data.bones:
        parent = name_compat(bone.parent.name) if bone.parent is not None else ''

        tail = bone.matrix_local.copy()
        tail.translation = bone.tail_local
        tail = global_matrix @ armature.matrix_world @ tail

        vx, vy, vz = tail.translation[:]

        mat = global_matrix @ armature.matrix_world @ bone.matrix_local

        if bone.parent:
            mat = global_matrix @ armature.matrix_world @ bone.matrix_local

        m = ""

        for xx in range(0, 4):
            for yy in range(0, 4):
                m += str(mat.row[xx][yy]) + ' '

        string = 'arm_bone %s %s ' % (name_compat(bone.name), parent)
        string += '%f %f %f ' % (vx, vy, vz)
        string += m + '\n'

        fw(string)

def write_actions(context, fw, convert_to_euler=False):
    fw('# Animation data\n')

    for key, action in bpy.data.actions.items():
        write_action(context, fw, key, action, convert_to_euler)

class VirtualKeyframe:
    def __init__(self, co, interp, hl, hr):
        self.co = co
        self.interpolation = interp
        self.handle_left = hl
        self.handle_right = hr

class VirtualFCurve:
    def __init__(self, dp, idx, kfs):
        self.data_path = dp
        self.array_index = idx
        self.keyframe_points = kfs

def write_action(context, fw, name, action, convert_to_euler=False):
    groups = {}

    def getOrCreate(key):
        if key in groups:
            return groups[key]
        l = []
        groups[key] = l
        return l

    for fc in action.fcurves:
        if fc.data_path.startswith('pose.bones["'):
            key = fc.data_path[12:]
            key = key[:key.index('"')]
            getOrCreate(key).append(fc)

    if not groups:
        return

    fw('an %s\n' % name)

    for key, group in groups.items():
        fw('ao %s\n' % name_compat(key))

        quat_curves = [fc for fc in group if fc.data_path.endswith('rotation_quaternion')]

        final_group = []

        if convert_to_euler and len(quat_curves) > 0:
            final_group.extend([fc for fc in group if not fc.data_path.endswith('rotation_quaternion')])

            qc_map = {fc.array_index: fc for fc in quat_curves}

            times = set()
            for fc in quat_curves:
                for kf in fc.keyframe_points:
                    times.add(kf.co[0])
            sorted_times = sorted(list(times))

            euler_kfs = [[], [], []]

            base_path = quat_curves[0].data_path.replace('rotation_quaternion', 'rotation_euler')

            for t in sorted_times:
                vals = []
                for i in range(4):
                    if i in qc_map:
                        vals.append(qc_map[i].evaluate(t))
                    else:
                        vals.append(0.0 if i > 0 else 1.0)

                q = mathutils.Quaternion(vals)
                eul = q.to_euler('XYZ')

                vals_prev = []
                vals_next = []
                for i in range(4):
                    if i in qc_map:
                        vals_prev.append(qc_map[i].evaluate(t - 1.0))
                        vals_next.append(qc_map[i].evaluate(t + 1.0))
                    else:
                        v = 0.0 if i > 0 else 1.0
                        vals_prev.append(v)
                        vals_next.append(v)

                q_prev = mathutils.Quaternion(vals_prev)
                q_next = mathutils.Quaternion(vals_next)

                eul_prev = q_prev.to_euler('XYZ')
                eul_next = q_next.to_euler('XYZ')

                eul_prev.make_compatible(eul)
                eul_next.make_compatible(eul)

                for axis in range(3):
                    co = mathutils.Vector((t, eul[axis]))
                    hl = mathutils.Vector((t - 1.0, eul_prev[axis]))
                    hr = mathutils.Vector((t + 1.0, eul_next[axis]))

                    vkf = VirtualKeyframe(co, 'BEZIER', hl, hr)
                    euler_kfs[axis].append(vkf)

            for axis in range(3):
                if euler_kfs[axis]:
                    vfc = VirtualFCurve(base_path, axis, euler_kfs[axis])
                    final_group.append(vfc)

        else:
            final_group = group

        for fcurve in final_group:
            data_path = fcurve.data_path
            index = fcurve.array_index
            length = len(fcurve.keyframe_points)
            dvalue = 0

            if data_path.endswith('location'):
                data_path = 'location'
            elif data_path.endswith('rotation_euler'):
                data_path = 'rotation'
            elif data_path.endswith('scale'):
                data_path = 'scale'
                dvalue = 1
            else:
                continue

            if length <= 0:
                continue

            all_default = True

            for keyframe in fcurve.keyframe_points:
                if keyframe.co[1] != dvalue:
                    all_default = False
                    break

            if all_default:
                continue

            fw('ag %s %d\n' % (data_path, index))

            last_frame = None

            for keyframe in fcurve.keyframe_points:
                if last_frame == keyframe.co[0]:
                    continue

                fw(stringify_keyframe(context, keyframe) + '\n')
                last_frame = keyframe.co[0]

def save(context, filepath, *, use_selection=True, include_keyframes=True, global_matrix=None, convert_to_euler=True, export_textures=False):
    if global_matrix is None:
        global_matrix = mathutils.Matrix.Rotation(math.radians(-90), 4, 'X')

    scene = context.scene

    if bpy.ops.object.mode_set.poll():
        bpy.ops.object.mode_set(mode='OBJECT')

    objects = context.selected_objects if use_selection else scene.objects

    write_file(context, filepath, objects, scene, global_matrix, include_keyframes, convert_to_euler, export_textures)

    return {'FINISHED'}

def write_file(context, filepath, objects, scene, EXPORT_GLOBAL_MATRIX, EXPORT_KEYFRAMES, convert_to_euler=True, export_textures=False):

    depsgraph = context.evaluated_depsgraph_get()

    def veckey3d(v):
        return round(v.x, 4), round(v.y, 4), round(v.z, 4)

    def veckey2d(v):
        return round(v[0], 5), round(v[1], 5)

    with open(filepath, "w", encoding="utf8", newline="\n") as f:
        fw = f.write

        fw('# Blender v%s Blockbuster OBJ File: %r\n' % (bpy.app.version_string, os.path.basename(bpy.data.filepath)))
        fw('# www.blender.org\n')

        totverts = totuvco = totno = 1
        face_vert_index = 1

        fw('# Armature data\n') # Force header

        written_armatures = set()

        def process_armature(arm_ob):
            if arm_ob.data.name in written_armatures:
                return
            print(f"DEBUG: Found Armature {arm_ob.name}")
            write_armature(fw, arm_ob, EXPORT_GLOBAL_MATRIX)
            written_armatures.add(arm_ob.data.name)

        def find_armature_parent(ob):
            curr = ob
            while curr:
                if curr.type == 'ARMATURE':
                    return curr
                curr = curr.parent
            return None

        for ob in objects:
            arm = find_armature_parent(ob)
            if arm:
                process_armature(arm)

        for i, ob_main in enumerate(objects):
            obs = [(ob_main, ob_main.matrix_world)]

            for ob, ob_mat in obs:
                uv_unique_count = no_unique_count = 0

                if ob.type == 'ARMATURE':
                    continue

                try:
                    ob_eval = ob.evaluated_get(depsgraph)
                    me = ob_eval.to_mesh()
                except RuntimeError:
                    me = None

                if me is None:
                    continue

                # Transform object's and global matrix and triangulate the mesh
                me.transform(EXPORT_GLOBAL_MATRIX @ ob_mat)
                mesh_triangulate(me)

                faceuv = len(me.uv_layers) > 0

                if faceuv:
                    uv_layer = me.uv_layers.active.data[:]

                me_verts = me.vertices[:]

                face_index_pairs = [(face, index) for index, face in enumerate(me.polygons)]

                if not (len(face_index_pairs) + len(me.vertices)):
                    ob_eval.to_mesh_clear()
                    continue

                if bpy.app.version < (4, 1, 0):
                    me.calc_normals_split()
                    get_normal = lambda l_idx: me.loops[l_idx].normal
                else:
                    get_normal = lambda l_idx: me.corner_normals[l_idx].vector

                loops = me.loops

                sort_func = lambda a: (a[0].material_index, a[0].use_smooth)
                face_index_pairs.sort(key=sort_func)
                del sort_func

                fw('# Mesh data\n')
                fw('o %s\n' % name_compat(ob.name))

                if ob.parent and ob.parent.type == 'ARMATURE':
                    fw('o_arm %s\n' % name_compat(ob.parent.data.name))

                for v in me_verts:
                    fw('v %.6f %.6f %.6f\n' % v.co[:])

                    for vgroup in v.groups:
                        fw('vw %s %.6f\n' % (name_compat(ob.vertex_groups[vgroup.group].name), vgroup.weight))

                if faceuv:
                    uv_face_mapping = [None] * len(face_index_pairs)

                    uv_dict = {}
                    uv_get = uv_dict.get
                    for f, f_index in face_index_pairs:
                        uv_ls = uv_face_mapping[f_index] = []
                        for uv_index, l_index in enumerate(f.loop_indices):
                            uv = uv_layer[l_index].uv
                            uv_key = loops[l_index].vertex_index, veckey2d(uv)

                            uv_val = uv_get(uv_key)
                            if uv_val is None:
                                uv_val = uv_dict[uv_key] = uv_unique_count
                                fw('vt %.4f %.4f\n' % uv[:])
                                uv_unique_count += 1
                            uv_ls.append(uv_val)

                    del uv_dict, uv, f_index, uv_index, uv_ls, uv_get, uv_key, uv_val

                no_key = no_val = None
                normals_to_idx = {}
                no_get = normals_to_idx.get
                loops_to_normals = [0] * len(loops)
                for f, f_index in face_index_pairs:
                    for l_idx in f.loop_indices:
                        no_key = veckey3d(get_normal(l_idx))
                        no_val = no_get(no_key)
                        if no_val is None:
                            no_val = normals_to_idx[no_key] = no_unique_count
                            fw('vn %.4f %.4f %.4f\n' % no_key)
                            no_unique_count += 1
                        loops_to_normals[l_idx] = no_val
                del normals_to_idx, no_get, no_key, no_val

                for f, f_index in face_index_pairs:
                    f_v = [(vi, me_verts[v_idx], l_idx) for vi, (v_idx, l_idx) in enumerate(zip(f.vertices, f.loop_indices))]

                    fw('f')
                    for vi, v, li in f_v:
                        uv_idx = totuvco + uv_face_mapping[f_index][vi] if faceuv else 0
                        norm_idx = totno + loops_to_normals[li]
                        fw(" %d/%d/%d" % (totverts + v.index, uv_idx, norm_idx))

                        face_vert_index += len(f_v)
                    fw('\n')

                totverts += len(me_verts)
                totuvco += uv_unique_count
                totno += no_unique_count

                ob_eval.to_mesh_clear()

        if EXPORT_KEYFRAMES:
            write_actions(context, fw, convert_to_euler)

    if export_textures:
        images = set()
        for ob in objects:
            if ob.type == 'MESH':
                for mat in ob.data.materials:
                    if mat and mat.use_nodes:
                        for node in mat.node_tree.nodes:
                            if node.type == 'BSDF_PRINCIPLED':
                                if node.inputs['Base Color'].is_linked:
                                    for link in node.inputs['Base Color'].links:
                                        if link.from_node.type == 'TEX_IMAGE':
                                            img = link.from_node.image
                                            if img:
                                                images.add(img)

        base_dir = os.path.dirname(filepath)
        for img in images:
            try:
                img_name = os.path.basename(img.filepath) if img.filepath else img.name + ".png"
                if not img_name:
                    img_name = "texture.png"

                dest_path = os.path.join(base_dir, img_name)

                src_path = bpy.path.abspath(img.filepath) if img.filepath else None
                if src_path and os.path.isfile(src_path):
                    if os.path.abspath(src_path) != os.path.abspath(dest_path):
                        shutil.copy2(src_path, dest_path)
                else:
                    img.save_render(dest_path)

            except Exception as e:
                print(f"Error exporting texture {img.name}: {e}")



class ImportBOBJ(Operator, ImportHelper):
    bl_idname = "import_scene.bobj"
    bl_label = "Import BOBJ"
    filename_ext = ".bobj"

    filter_glob: StringProperty(
        default="*.bobj",
        options={'HIDDEN'},
        maxlen=255,
    )

    import_force_xyz: BoolProperty(
        name="Import Rotation XYZ",
        description="Import raw XYZ rotation (skips orientation correction)",
        default=False,
    )

    def execute(self, context):
        return read_bobj(context, self.filepath, import_force_xyz=self.import_force_xyz)

def read_bobj(context, filepath, import_force_xyz=False):
    scene = context.scene
    fps = scene.render.fps

    all_uvs = []
    all_normals = []

    meshes = []
    armatures = {} # name -> { 'action' name, 'bones' [] }
    actions = {} # name -> { bone_name: { data_path: { index: [keyframes] } } }

    current_mesh_name = None
    current_verts = []
    current_faces = []
    current_weights = {}
    current_parent_arm = None

    current_armature_name = None
    current_action_name = None
    current_anim_obj = None
    current_keyframes = None

    vertex_offset = 0

    if bpy.ops.object.select_all.poll():
        bpy.ops.object.select_all(action='DESELECT')

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            tag = parts[0]

            if tag == 'o':
                if current_mesh_name:
                    meshes.append({
                        'name': current_mesh_name,
                        'verts': current_verts,
                        'faces': current_faces,
                        'weights': current_weights,
                        'parent': current_parent_arm
                    })
                    vertex_offset += len(current_verts)

                current_mesh_name = parts[1]
                current_verts = []
                current_faces = []
                current_weights = {}
                current_parent_arm = None

            elif tag == 'o_arm':
                current_parent_arm = parts[1]

            elif tag == 'v':
                current_verts.append((float(parts[1]), float(parts[2]), float(parts[3])))

            elif tag == 'vw':
                group = parts[1]
                weight = float(parts[2])
                v_idx = len(current_verts) - 1
                if v_idx >= 0:
                    if v_idx not in current_weights:
                        current_weights[v_idx] = []
                    current_weights[v_idx].append((group, weight))

            elif tag == 'vt':
                all_uvs.append((float(parts[1]), float(parts[2])))

            elif tag == 'vn':
                all_normals.append((float(parts[1]), float(parts[2]), float(parts[3])))

            elif tag == 'f':
                face_indices = []
                for p in parts[1:]:
                    sub = p.split('/')
                    v_g = int(sub[0])
                    vt_g = int(sub[1]) if len(sub) > 1 and sub[1] else 0
                    vn_g = int(sub[2]) if len(sub) > 2 and sub[2] else 0

                    v_l = v_g - 1 - vertex_offset
                    face_indices.append((v_l, vt_g, vn_g))
                current_faces.append(face_indices)

            elif tag == 'arm_name':
                current_armature_name = parts[1]
                armatures[current_armature_name] = {'action': None, 'bones': []}

            elif tag == 'arm_action':
                if current_armature_name:
                    armatures[current_armature_name]['action'] = parts[1]

            elif tag == 'arm_bone':
                if current_armature_name:
                    if len(parts) == 22:
                        name = parts[1]
                        parent = parts[2]
                        tail_idx = 3
                    elif len(parts) == 21:
                        name = parts[1]
                        parent = None
                        tail_idx = 2
                    else:
                        print(f"Warning: Malformed arm_bone line: {line}")
                        continue

                    tail = (float(parts[tail_idx]), float(parts[tail_idx+1]), float(parts[tail_idx+2]))
                    mat_vals = [float(x) for x in parts[tail_idx+3:]]

                    armatures[current_armature_name]['bones'].append({
                        'name': name,
                        'parent': parent,
                        'tail': tail,
                        'matrix': mat_vals
                    })

            elif tag == 'an':
                current_action_name = parts[1]
                actions[current_action_name] = {}

            elif tag == 'ao':
                current_anim_obj = parts[1]
                if current_action_name:
                    if current_anim_obj not in actions[current_action_name]:
                        actions[current_action_name][current_anim_obj] = {}

            elif tag == 'ag':
                dp = parts[1]
                idx = int(parts[2])
                if current_action_name and current_anim_obj:
                    if dp not in actions[current_action_name][current_anim_obj]:
                        actions[current_action_name][current_anim_obj][dp] = {}
                    actions[current_action_name][current_anim_obj][dp][idx] = []
                    current_keyframes = actions[current_action_name][current_anim_obj][dp][idx]
                else:
                    current_keyframes = None

            elif tag == 'kf':
                if current_keyframes is not None:
                    current_keyframes.append((
                        float(parts[1]), float(parts[2]), parts[3],
                        float(parts[4]), float(parts[5]),
                        float(parts[6]), float(parts[7])
                    ))

    if current_mesh_name:
        meshes.append({
            'name': current_mesh_name,
            'verts': current_verts,
            'faces': current_faces,
            'weights': current_weights,
            'parent': current_parent_arm
        })


    arm_objs = {}

    for arm_name, arm_data in armatures.items():
        amt = bpy.data.armatures.new(arm_name)
        obj = bpy.data.objects.new(arm_name, amt)
        context.collection.objects.link(obj)

        context.view_layer.objects.active = obj
        obj.select_set(True)
        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode='EDIT')

        edit_bones = amt.edit_bones

        for b_data in arm_data['bones']:
            edit_bones.new(b_data['name'])

        for b_data in arm_data['bones']:
            if b_data['parent'] and b_data['parent'] in edit_bones:
                edit_bones[b_data['name']].parent = edit_bones[b_data['parent']]

        for b_data in arm_data['bones']:
            bone = edit_bones[b_data['name']]
            mat_rows = [b_data['matrix'][i:i+4] for i in range(0, 16, 4)]
            bone.matrix = mathutils.Matrix(mat_rows)
            bone.tail = mathutils.Vector(b_data['tail'])

        if bpy.ops.object.mode_set.poll():
            bpy.ops.object.mode_set(mode='OBJECT')

        for pb in obj.pose.bones:
            pb.rotation_mode = 'XYZ'

        if not import_force_xyz:
            obj.rotation_euler[0] = math.radians(90)

        arm_objs[arm_name] = obj

    for m_data in meshes:
        mesh = bpy.data.meshes.new(m_data['name'])
        obj = bpy.data.objects.new(m_data['name'], mesh)
        context.collection.objects.link(obj)

        face_v_indices = [[idx[0] for idx in f] for f in m_data['faces']]
        mesh.from_pydata(m_data['verts'], [], face_v_indices)

        if all_uvs and any(f[0][1] > 0 for f in m_data['faces']):
            uv_layer = mesh.uv_layers.new()
            for i, poly in enumerate(mesh.polygons):
                for j, loop_idx in enumerate(poly.loop_indices):
                    vt_idx = m_data['faces'][i][j][1]
                    if vt_idx > 0:
                        uv_layer.data[loop_idx].uv = all_uvs[vt_idx - 1]

        if all_normals and any(f[0][2] > 0 for f in m_data['faces']):
            loop_normals = []
            for i, poly in enumerate(mesh.polygons):
                for j, loop_idx in enumerate(poly.loop_indices):
                    vn_idx = m_data['faces'][i][j][2]
                    if vn_idx > 0:
                        loop_normals.append(all_normals[vn_idx - 1])
                    else:
                        loop_normals.append((0, 0, 1))

            mesh.normals_split_custom_set(loop_normals)

            mesh.polygons.foreach_set("use_smooth", [False] * len(mesh.polygons))

            if bpy.app.version < (4, 1, 0):
                mesh.use_auto_smooth = False
            else:
                bpy.ops.object.select_all(action='DESELECT')
                context.view_layer.objects.active = obj
                obj.select_set(True)
                bpy.ops.object.shade_flat()

        for v_idx, w_list in m_data['weights'].items():
            for grp, w in w_list:
                vg = obj.vertex_groups.get(grp)
                if not vg:
                    vg = obj.vertex_groups.new(name=grp)
                vg.add([v_idx], w, 'REPLACE')

        if m_data['parent'] and m_data['parent'] in arm_objs:
            par = arm_objs[m_data['parent']]
            obj.parent = par
            mod = obj.modifiers.new(name="Armature", type='ARMATURE')
            mod.object = par
        else:
            if not import_force_xyz:
                obj.rotation_euler[0] = math.radians(90)

    for act_name, act_data in actions.items():
        action = bpy.data.actions.new(act_name)

        for bone_name, curves in act_data.items():
            for dp_key, indices in curves.items():
                real_dp = ""
                sanitized_name = bone_name.replace(' ', '_')
                if dp_key == 'location':
                    real_dp = 'pose.bones["%s"].location' % sanitized_name
                elif dp_key == 'rotation':
                    real_dp = 'pose.bones["%s"].rotation_euler' % sanitized_name
                elif dp_key == 'scale':
                    real_dp = 'pose.bones["%s"].scale' % sanitized_name
                else:
                    continue

                for idx, kfs in indices.items():
                    fc = action.fcurves.new(real_dp, index=idx)
                    fc.keyframe_points.add(len(kfs))

                    for i, kf in enumerate(kfs):
                        kp = fc.keyframe_points[i]

                        factor = fps / 20.0

                        t = kf[0] * factor
                        v = kf[1]
                        interp = kf[2]
                        hl_t = kf[3] * factor
                        hl_v = kf[4]
                        hr_t = kf[5] * factor
                        hr_v = kf[6]

                        kp.co = (t, v)
                        kp.interpolation = interp

                        kp.handle_left_type = 'FREE'
                        kp.handle_right_type = 'FREE'

                        kp.handle_left = (hl_t, hl_v)
                        kp.handle_right = (hr_t, hr_v)

        for arm_name, arm_d in armatures.items():
            if arm_d['action'] == act_name:
                if arm_name in arm_objs:
                    ao = arm_objs[arm_name]
                    if not ao.animation_data:
                        ao.animation_data_create()
                    ao.animation_data.action = action

    return {'FINISHED'}



def menu_func_import(self, context):
    self.layout.operator(ImportBOBJ.bl_idname, text="Blockbuster OBJ (.bobj)")

def menu_func_export(self, context):
    self.layout.operator(ExportBOBJ.bl_idname, text="Blockbuster OBJ (.bobj)")

def register():
    bpy.utils.register_class(ImportBOBJ)
    bpy.utils.register_class(ExportBOBJ)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.append(menu_func_export)

def unregister():
    bpy.utils.unregister_class(ImportBOBJ)
    bpy.utils.unregister_class(ExportBOBJ)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    bpy.types.TOPBAR_MT_file_export.remove(menu_func_export)

if __name__ == "__main__":
    register()