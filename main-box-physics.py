import blenderproc as bproc
import argparse
import numpy as np
import os
from glob import glob
import random
import h5py
import cv2
# from blenderproc.scripts.saveAsImg import convert_hdf
from blenderproc.scripts.visHdf5Files import vis_data
'''
run it like the following

   blenderproc run main.py scene.blend ./output
   
  or 
  
   blenderproc run main.py ./output 
'''


parser = argparse.ArgumentParser()
#parser.add_argument('scene', nargs='?', default="examples/basics/semantic_segmentation/scene.blend", help="Path to the scene.obj file")
parser.add_argument('--output-dir', nargs='?', default="output", help="Path to where the final files, will be saved")
parser.add_argument("--data-path", type=str, default="/media/ws1/Data3/datasets/sps_synthetic", help="parent dir for dataset")
parser.add_argument("--id", type=int, default=0, help="prefix for saving ouput")
args = parser.parse_args()

subdirs = ["rgb", "depth", "gt"]
for dir in subdirs:
    os.makedirs(os.path.join(args.data_path, dir), exist_ok=True)
bproc.init()

# load the objects into the scene
#objs = bproc.loader.load_blend(args.scene)

table = bproc.loader.load_obj('./assets/chute.obj')[0]
table.set_location([-3,0,0])
# need collision margin to prevent thin objects from "falling" into the table
# also make sure to set collision_shape of thin objects to MESH
table.enable_rigidbody(active=False, collision_margin=0.0005, collision_shape="MESH")

# wall1 = bproc.loader.load_obj('./assets/wall.obj')[0]
# wall1.set_location([-45,0,55])
# wall1.set_rotation_euler([np.pi/2, 0, 0])
# wall1.enable_rigidbody(active=False, collision_margin=0.0005, collision_shape="MESH")

# light panel with emissive property light_surfaceq
lights = bproc.loader.load_obj('./assets/light-panel.obj')[0]
lights.set_location([3,3,30])
bproc.lighting.light_surface([lights], 30) #second arg is emission strength


# get list of available objects
assets = glob("./assets/sps/*.obj")
print("found the following assets: {}".format(assets))

# min and max num objects
min_objs = 10
max_objs = 20
n_objs = random.randint(min_objs, max_objs)
# n_objs = 5

obj_list = random.choices(assets, k=n_objs)

print(" The following objects will be used for sim: {}".format(obj_list))

objs = []
for i in range(n_objs):
    obj_name = os.path.basename(obj_list[i]).replace(".obj", "")
    obj = bproc.loader.load_obj(obj_list[i])[0]
    #todo: find good locations
    obj.set_location(np.random.uniform([0, 0, 5], [4, 2, 12]))
    if obj_name == "box2":
        obj.set_rotation_euler(np.random.uniform([0, 0, 0], [np.pi / 4, 0, np.pi / 20]))
        obj.enable_rigidbody(active=True)
    else:
        obj.enable_rigidbody(active=True, collision_shape='MESH')
    objs.append(obj)
    

# box1 = bproc.loader.load_obj('./Box/box2.obj')[0]
# box1.set_location([0,0,5])
# #box1.set_rotation_euler([np.pi / 4, 0, np.pi / 20])
# box1.set_rotation_euler(np.random.uniform([0, 0, 0], [np.pi / 4, 0, np.pi / 20]))
# box1.enable_rigidbody(active=True)

# box2 = bproc.loader.load_obj('./Box/box2.obj')[0]
# box2.set_location([2, 2, 12])  
# #box2.set_rotation_euler(np.random.uniform([0, 0, 0], [np.pi / 4, np.pi / 4, np.pi / 4]))
# box2.enable_rigidbody(active=True)

# box3 = bproc.loader.load_obj('./Box/box2.obj')[0]
# box3.set_location([-2, -2, 9])  
# #box3.set_rotation_euler(np.random.uniform([0, 0, 0], [np.pi / 4, np.pi / 4, np.pi / 4]))
# box3.enable_rigidbody(active=True)


# box4 = bproc.loader.load_obj('./Box/box2.obj')[0]
# box4.set_location([0, -2, 14])  
# #box3.set_rotation_euler(np.random.uniform([0, 0, 0], [np.pi / 4, np.pi / 4, np.pi / 4]))
# box4.enable_rigidbody(active=True)

# box5 = bproc.loader.load_obj('./Box/box2.obj')[0]
# box5.set_location([3, 0, 8])  
# #box3.set_rotation_euler(np.random.uniform([0, 0, 0], [np.pi / 4, np.pi / 4, np.pi / 4]))
# box5.enable_rigidbody(active=True)


# # load envelopes
# # for flat objects, set collision_shape to 'MESH' prevented object from going into the floor

# env1 = bproc.loader.load_obj('./amazon.obj')[0]
# env1.set_location([3,1,6])
# env1.enable_rigidbody(active=True, collision_shape='MESH')
# #env1.enable_rigidbody(active=True)

# env2 = bproc.loader.load_obj('./prime.obj')[0]
# env2.set_location([-1,-3,11])
# env2.enable_rigidbody(active=True, collision_shape='MESH')
# #env2.enable_rigidbody(active=True)


# define a light and set its location and energy level

# area light is not good, use emissive 'light panel' instead above
'''
light = bproc.types.Light()
light.set_type('AREA')
light.set_location([2, 0, 15])
light.set_energy(3000)
'''

# define the camera intrinsics
bproc.camera.set_resolution(640, 480)

# LSC: set fixed camera
# Set intrinsics via K matrix


bproc.camera.set_intrinsics_from_K_matrix(
    [[615.5454, 0.0, 309.1487],
     [0.0, 615.5802, 220.9723],
     [0.0, 0.0, 1.0]], 640, 480
)

# Set camera pose via cam-to-world transformation matrix
# trial and error: rotation matrix with [1 -1 -1] diagonal works (rotate -90 about x )

# strange... when z is 25 or larger, depth disappears
cam2world = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 18],
    [0, 0, 0, 1]
])
cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(cam2world, ["X", "-Y", "-Z"])

bproc.camera.add_camera_pose(cam2world)


# add a second camera [ABANDONED]
# (1) larger focal length compared to the first
# (2) placement is farther away from objects
# HOWEVER, experienced several issues and decided to abandon it
# issue1: depth info is lost due to the camera being farther away
# issue2: got an error while extracting the seg_map from hdf5
'''
bproc.camera.set_intrinsics_from_K_matrix(
    [[1230, 0.0, 309.1487],
     [0.0, 1230, 220.9723],
     [0.0, 0.0, 1.0]], 640, 480
)
cam2world = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 35],
    [0, 0, 0, 1]
])
cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(cam2world, ["X", "-Y", "-Z"])

bproc.camera.add_camera_pose(cam2world)
'''

#    bproc.camera.add_camera_pose(bproc.math.build_transformation_mat([0, -3.5, 10], [0.5, 0, 0]))

# sanity check to make sure object is above the table
#bproc.camera.add_camera_pose(bproc.math.build_transformation_mat([0, -15, 3], [1.57, 0, 0]))

# activate depth rendering
#bproc.renderer.render_depth(True)
bproc.renderer.enable_depth_output(activate_antialiasing=False)

# Run the simulation and fix the poses of the spheres at the end
bproc.object.simulate_physics_and_fix_final_poses(min_simulation_time=4, max_simulation_time=20, check_object_interval=1)
# bproc.object.simulate_physics(min_simulation_time=4, max_simulation_time=20, check_object_interval=1)
# bproc.utility.set_keyframe_render_interval(frame_end=90)
#bproc.utility.set_keyframe_render_interval(frame_end = 1)
# render the whole pipeline
data = bproc.renderer.render()

# Render segmentation masks (per class and per instance)
data.update(bproc.renderer.render_segmap(map_by=["instance", "name"]))

# seg_data = bproc.renderer.render_segmap(map_by=["instance", "class", "name"])

# write the data to a .hdf5 container
bproc.writer.write_hdf5(args.output_dir, data)
# bproc.writer.write_gif_animation(args.output_dir, data)
with h5py.File(os.path.join(args.output_dir, "0.hdf5"), 'r') as data:
    keys = [key for key in data.keys()]
    for key in keys:
        val = np.array(data[key])
        if np.issubdtype(val.dtype, np.string_) or len(val.shape) == 1:
            pass  # metadata
        else:
            print("key: {}  {} {}".format(key, val.shape, val.dtype.name))

            if val.shape[0] != 2:
                # mono image
                if key == "colors":
                    file_path = os.path.join(args.data_path, "rgb", str(args.id) + ".png")
                elif key == "depth":
                    file_path = os.path.join(args.data_path, "depth", str(args.id) + ".png")
                elif key == "instance_segmaps":
                    file_path = os.path.join(args.data_path, "gt", str(args.id) + ".png")
                vis_data(key, val, None, "", save_to_file=file_path)
            else:
                # stereo image
                for image_index, image_value in enumerate(val):
                    file_path = '{}_{}.png'.format(key, image_index)
                    vis_data(key, val, None, "", save_to_file=file_path)

