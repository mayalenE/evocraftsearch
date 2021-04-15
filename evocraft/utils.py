import grpc
from nbt import nbt
import json
import torch
import os
from collections import OrderedDict

from evocraftsearch.evocraft import minecraft_pb2_grpc
from evocraftsearch.evocraft.minecraft_pb2 import *
channel = grpc.insecure_channel('localhost:5001') #WORLD ORIGIN: (0,4,0)
client = minecraft_pb2_grpc.MinecraftServiceStub(channel)


def save_arena(outnbt_filepath, arena_bbox=(0,4,0,20,20,20), name='', author=''):
    arena_blocks = client.readCube(Cube(
        min=Point(x=arena_bbox[0], y=arena_bbox[1], z=arena_bbox[2]),
        max=Point(x=arena_bbox[0]+arena_bbox[3]-1, y=arena_bbox[1]+arena_bbox[4]-1, z=arena_bbox[2]+arena_bbox[5]-1)
    ))

    nbtfile = nbt.NBTFile()

    # fill main tag name
    nbtfile.name = name

    # fill arena size
    size = nbt.TAG_List(name="size", type=nbt.TAG_Int)
    size.append(nbt.TAG_Int(arena_bbox[3]))
    size.append(nbt.TAG_Int(arena_bbox[4]))
    size.append(nbt.TAG_Int(arena_bbox[5]))
    nbtfile.tags.append(size)

    # fill entities present in the arena (here none)
    #nbtfile.tags.append(nbt.TAG_List(name="entities"))

    # fill blocks
    blocks = nbt.TAG_List(name="blocks", type=nbt.TAG_Compound)
    arena_block_types = []
    for block in arena_blocks.blocks:
        if block.type not in arena_block_types:
            block_idx = len(arena_block_types)
            arena_block_types.append(block.type)
        else:
            block_idx = arena_block_types.index(block.type)
        block_pos = nbt.TAG_List(name="pos", type=nbt.TAG_Int)
        block_pos.append(nbt.TAG_Int(block.position.x - arena_bbox[0]))
        block_pos.append(nbt.TAG_Int(block.position.y - arena_bbox[1]))
        block_pos.append(nbt.TAG_Int(block.position.z - arena_bbox[2]))
        block_nbt = nbt.TAG_Compound()
        block_nbt.tags.append(block_pos)
        block_state = nbt.TAG_Int(block_idx)
        block_state.name = "state"
        block_nbt.tags.append(block_state)
        blocks.append(block_nbt)
    nbtfile.tags.append(blocks)

    # fill author
    nbtfile.tags.append(nbt.TAG_String(name="author", value=author))

    # fill palette
    palette = nbt.TAG_List(name="palette", type=nbt.TAG_Compound)
    for block_type in arena_block_types:
        block_type_nbt = nbt.TAG_Compound()
        block_name = nbt.TAG_String(name="Name", value='minecraft:' + BlockType.keys()[block_type].lower()) #'Properties': nbt.TAG_Compound({'facing': block.orientation.lower()})
        block_type_nbt.tags.append(block_name)
        palette.append(block_type_nbt)
    nbtfile.tags.append(palette)

    nbtfile.write_file(outnbt_filepath)

def load_arena(nbt_filepath, arena_bbox=(0,4,0,20,20,20)):
    """
    Loads an arena saved in nbt file and draws it into MineCraft world
    :param nbt_filepath to load from
    :param arena_bbox = (o_x, o_y, o_z, w, h, d)
    """
    assert os.path.isfile(nbt_filepath) and '.nbt' in nbt_filepath
    nbtfile = nbt.NBTFile(nbt_filepath, 'rb')

    # draw
    min_x, max_x = (nbtfile['blocks'][0]['pos'][0].value, nbtfile['blocks'][0]['pos'][0].value)
    min_y, max_y = (nbtfile['blocks'][0]['pos'][1].value, nbtfile['blocks'][0]['pos'][1].value)
    min_z, max_z = (nbtfile['blocks'][0]['pos'][2].value, nbtfile['blocks'][0]['pos'][2].value)

    block_types = nbtfile['palette']

    all_pos = []
    all_orientations = []
    all_types = []
    for block in nbtfile['blocks']:
        cur_x, cur_y, cur_z = block['pos'][0].value, block['pos'][1].value, block['pos'][2].value
        min_x = min(min_x, cur_x)
        max_x = max(max_x, cur_x)
        min_y = min(min_y, cur_y)
        max_y = max(max_y, cur_y)
        min_z = min(min_z, cur_z)
        max_z = max(max_z, cur_z)
        all_pos.append((cur_x, cur_y, cur_z))

        cur_block_type = block_types[block['state'].value]
        cur_orientation = 'NORTH' #orientation by default
        if 'Properties' in cur_block_type.keys():
            if 'facing' in cur_block_type['Properties'].keys():
                cur_orientation = cur_block_type['Properties']['facing'].value.upper()
        all_orientations.append(cur_orientation)

        cur_type = cur_block_type['Name'].value.split(":")[-1].upper()
        all_types.append(cur_type)


    blocks = []
    for block_idx, block_pos in enumerate(all_pos):
        # translate block pos to fit in area
        world_x = block_pos[0] - min_x + arena_bbox[0]
        world_y = block_pos[1] - min_y + arena_bbox[1]
        world_z = block_pos[2] - min_z + arena_bbox[2]

        if (world_x < arena_bbox[0]+arena_bbox[3]) and (world_y < arena_bbox[1]+arena_bbox[4]) and (world_z < arena_bbox[2]+arena_bbox[5]):
            cur_pos = Point(x=world_x, y=world_y, z=world_z)
            cur_block = Block(position=cur_pos, type=all_types[block_idx], orientation=all_orientations[block_idx])
            blocks.append(cur_block)

    # Clear the necessary working area
    client.fillCube(FillCubeRequest(
        cube=Cube(
            min=Point(x=arena_bbox[0], y=arena_bbox[1], z=arena_bbox[2]),
            max=Point(x=arena_bbox[0]+arena_bbox[3]-1, y=arena_bbox[1]+arena_bbox[4]-1, z=arena_bbox[2]+arena_bbox[5]-1)
        ),
        type=AIR
    ))

    # Draw the loaded blocks
    client.spawnBlocks(Blocks(blocks=blocks))


def get_minecraft_color_list(block_list):
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'palette.json'), 'rb') as f:
        json_data = json.load(f)
    color_dict = OrderedDict.fromkeys(block_list)
    for block_data in json_data:
        if block_data["mode"] == "block" and block_data["material"].upper() in block_list:
            color_dict[block_data["material"].upper()] = torch.tensor(block_data["top_color"]).float() / 255.0
    return list(color_dict.values())