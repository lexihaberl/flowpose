import blenderproc as bproc
import os
import numpy as np
import json
from pathlib import Path


def load_objects(path, category_names, num_meshes_per_category=1):
    objects = []
    for category in category_names:
        meshes_loaded_per_category = 0
        object_folder = Path(path, category)
        for file in sorted(os.listdir(object_folder)):
            if meshes_loaded_per_category >= num_meshes_per_category:
                break
            if file.endswith(".obj"):
                object_name = file.split(".")[0]
                obj = bproc.loader.load_obj(str(object_folder / file))
                if len(obj) != 1:
                    raise ValueError(
                        f"I thought it only loads one mesh in this case... but it loaded {len(obj)} meshes"
                    )

                obj[0].hide(True)
                obj[0].set_name(object_name)
                obj[0].set_cp("category_id", 0)
                objects.append(obj[0])
                meshes_loaded_per_category += 1
    return objects


def create_room():
    room_planes = [
        bproc.object.create_primitive("PLANE", scale=[2, 2, 1]),
        bproc.object.create_primitive(
            "PLANE", scale=[2, 2, 1], location=[0, -2, 2], rotation=[-1.570796, 0, 0]
        ),
        bproc.object.create_primitive(
            "PLANE", scale=[2, 2, 1], location=[0, 2, 2], rotation=[1.570796, 0, 0]
        ),
        bproc.object.create_primitive(
            "PLANE", scale=[2, 2, 1], location=[2, 0, 2], rotation=[0, -1.570796, 0]
        ),
        bproc.object.create_primitive(
            "PLANE", scale=[2, 2, 1], location=[-2, 0, 2], rotation=[0, 1.570796, 0]
        ),
    ]
    for plane in room_planes:
        plane.enable_rigidbody(
            False,
            collision_shape="BOX",
            friction=100.0,
            linear_damping=0.99,
            angular_damping=0.99,
        )
        plane.set_cp("category_id", 1000)
    return room_planes


def create_lighting():
    light_plane = bproc.object.create_primitive(
        "PLANE", scale=[3, 3, 1], location=[0, 0, 10]
    )
    light_plane.set_name("light_plane")
    light_plane_material = bproc.material.create("light_material")
    light_plane_material.make_emissive(
        emission_strength=10,
        emission_color=np.random.uniform([0.5, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]),
    )
    light_plane.replace_materials(light_plane_material)
    light_plane.set_cp("category_id", 2000)
    return light_plane


def create_table(table_mesh_path=Path("data", "table_mesh", "table.obj")):
    tables = bproc.loader.load_obj(str(table_mesh_path))
    if len(tables) != 1:
        print(
            "I thought it only loads one mesh in this case... but it loaded {} meshes".format(
                len(tables)
            )
        )
    for table in tables:
        print(table.get_name())
        table.set_cp("category_id", 3000)
        table.enable_rigidbody(
            False,
            collision_shape="BOX",
            friction=100.0,
            linear_damping=0.99,
            angular_damping=0.99,
        )
        table.set_location([0, 0, 0.3])
    return tables


def get_table_surface(tables, table_name):
    for table in tables:
        print(table.get_name())
        if table.get_name() == table_name:
            return table
    print("Table not found")
    return None


def sample_initial_pose(obj):
    obj.set_location(
        bproc.sampler.upper_region(
            objects_to_sample_on=table_surface,
            min_height=1,
            max_height=4,
            face_sample_range=[0.2, 0.8],
        )
    )


def place_object(object, surface):
    placed_objects = bproc.object.sample_poses_on_surface(
        objects_to_sample=object,
        surface=surface,
        sample_pose_func=sample_initial_pose,
        min_distance=0.01,
        max_distance=0.2,
    )
    assert len(placed_objects) == 1
    return placed_objects[0]


camera_width = 640
camera_height = 480
camera_intrinsics_sasha = np.array(
    [
        [538.391033533567, 0.0, 315.3074696331638],
        [0.0, 538.085452058436, 233.0483557773859],
        [0.0, 0.0, 1.0],
    ]
)
output_dir = Path("output", "dataset_rendered")
mesh_dir = Path("data", "housecat6d_meshes")
cc_textures_path = Path("data", "cc_textures")
if os.path.exists(output_dir):
    raise ValueError("Output directory already exists")

bproc.init()
bproc.camera.set_intrinsics_from_K_matrix(
    camera_intrinsics_sasha, camera_width, camera_height
)


objects = load_objects(mesh_dir, ["shoe", "cutlery"], 2000)
room_planes = create_room()
light_plane = create_lighting()
cc_textures = bproc.loader.load_ccmaterials(cc_textures_path)
tables = create_table()
original_table_material = tables[0].get_materials()[0]
table_surface = get_table_surface(tables, "Cube")

partitions_radius = 4
partitions_elevation = 4
partitions_azimuth = 4
# only look at objects from 'above' (we assume the object is placed on a plane e.g. table)
radius_range = [0.3, 0.9]
elevation_angle_range = [0, 90]
azimuth_angle_range = [-180, 180]

radius_diff = radius_range[1] - radius_range[0]
elevation_angle_diff = elevation_angle_range[1] - elevation_angle_range[0]
azimuth_angle_diff = azimuth_angle_range[1] - azimuth_angle_range[0]

bproc.renderer.enable_depth_output(activate_antialiasing=False)
bproc.renderer.enable_segmentation_output(map_by=["name", "instance"])

for obj in objects:
    random_table_texture = np.random.choice(cc_textures)
    random_room_texture = np.random.choice(cc_textures)
    for table_obj in tables:
        # if uniform < 0.3 -> use original texture
        if np.random.uniform() < 0.3:
            table_obj.replace_materials(original_table_material)
        else:
            table_obj.replace_materials(random_table_texture)
    for room_obj in room_planes:
        room_obj.replace_materials(random_room_texture)
    obj = place_object([obj], table_surface)
    poses = {}
    bproc.utility.reset_keyframes()
    obj.hide(False)
    light_point = bproc.types.Light()
    light_point.set_energy(25)
    light_point.set_color(np.random.uniform([0.5, 0.5, 0.5], [1, 1, 1]))
    location = bproc.sampler.shell(
        center=[0, 0, 0.5],
        radius_min=0.5,
        radius_max=1.5,
        elevation_min=10,
        elevation_max=89,
        uniform_volume=False,
    )
    light_point.set_location(location)
    obj_pose = obj.get_local2world_mat()
    poses[obj.get_name()] = obj_pose.tolist()

    for k in range(partitions_azimuth):
        for j in range(partitions_elevation):
            for i in range(partitions_radius):
                radius_min = radius_range[0] + i * radius_diff / partitions_radius
                radius_max = radius_range[0] + (i + 1) * radius_diff / partitions_radius
                elevation_min = (
                    elevation_angle_range[0]
                    + j * elevation_angle_diff / partitions_elevation
                )
                elevation_max = (
                    elevation_angle_range[0]
                    + (j + 1) * elevation_angle_diff / partitions_elevation
                )
                azimuth_min = (
                    azimuth_angle_range[0] + k * azimuth_angle_diff / partitions_azimuth
                )
                azimuth_max = (
                    azimuth_angle_range[0]
                    + (k + 1) * azimuth_angle_diff / partitions_azimuth
                )

                camera_center = obj.get_origin()
                camera_location = bproc.sampler.shell(
                    camera_center,
                    radius_min,
                    radius_max,
                    elevation_min,
                    elevation_max,
                    azimuth_min,
                    azimuth_max,
                    uniform_volume=False,
                )
                point_of_interest = bproc.object.compute_poi([obj])
                rotation_matrix = bproc.camera.rotation_from_forward_vec(
                    point_of_interest - camera_location
                )  # inplane_rot=np.random.uniform(-0.7854, 0.7854))
                cam2world_matrix = bproc.math.build_transformation_mat(
                    camera_location, rotation_matrix
                )
                frame_idx = bproc.camera.add_camera_pose(cam2world_matrix)
                poses["cam_" + str(frame_idx)] = cam2world_matrix.tolist()

    data = bproc.renderer.render()
    obj_category = obj.get_name().split("-")[0]
    output_path = output_dir / obj_category / obj.get_name()
    bproc.writer.write_hdf5(output_path, data, append_to_existing_output=True)
    json.dump(poses, open(output_path / "poses.json", "w"))
    obj.hide(True)
    obj.clear_materials()
    obj.disable_rigidbody()
