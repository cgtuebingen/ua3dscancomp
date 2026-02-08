import sys
import bpy

from mathutils import Vector

infile = sys.argv[4]
outfile = sys.argv[5]

# remove all existing objects from the scene
for item in bpy.data.objects:
    bpy.data.objects.remove(item)

bpy.ops.import_scene.gltf(filepath=infile)

# deselect them all
bpy.ops.object.select_all(action='DESELECT')

for object in bpy.data.objects:
    # remove if no mesh
    if (object.type != "MESH") or (object not in bpy.context.visible_objects):
        bpy.data.objects.remove(object)
    else:
        # set object active
        bpy.context.view_layer.objects.active = object

        # remove all modifiers
        for m in object.modifiers:
            #try:
            #    bpy.ops.object.modifier_apply(modifier=m.name)
            #except:
            #    print('modifier', m.name, 'not applied')
            bpy.ops.object.modifier_remove(modifier=m.name)

        # apply all transformations
        bpy.ops.object.transform_apply()

# make sure there is a primary object selected. so it can assign its name as the name of the joined mesh
bpy.context.view_layer.objects.active = bpy.data.objects[0]

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.join()

# after joining, once we apply all the transformation to the join one
bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

mesh_active = bpy.context.view_layer.objects.active

# normalize after transform
X = mesh_active.dimensions.x
Y = mesh_active.dimensions.y
Z = mesh_active.dimensions.z

max_dim = max(X, Y, Z)
tolerance = 0.0001

corner = Vector(mesh_active.bound_box[0])
translation = (-corner.x - X / 2, -corner.y - Y / 2, -corner.z - Z / 2)

bpy.ops.transform.translate(value=translation)
bpy.ops.object.transform_apply(location=True, rotation=False, scale=False)


scale_val = (2 / (max_dim + tolerance), 2 / (max_dim + tolerance), 2 / (max_dim + tolerance))
bpy.ops.transform.resize(value=scale_val)
bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

# distance should be smaller than 2/128 after normalization
bpy.ops.object.editmode_toggle()
bpy.ops.mesh.remove_doubles(threshold=0.0001)
bpy.ops.object.editmode_toggle()

# export excluding the animation files
bpy.ops.export_scene.gltf(filepath=outfile, use_selection=True, export_apply=True, export_normals=False, export_texcoords=False, export_materials='NONE',
                          export_image_format='NONE', export_colors=False, export_animations=False, export_skins=False, export_morph=False)
# delete all selected objects
bpy.ops.object.delete()

# bpy.ops.console.clear()

# keep blender from loading extra arguments as python scripts / blend files
exit(0)
