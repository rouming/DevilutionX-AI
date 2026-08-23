"""
devilutionx_generate.py - Generator of the devilutionx.py module
"""
import dbg2numpy

# Variables which should be exported
DEVILUTIONX_VARS = [
    "devilution::shared::input_queue",
    "devilution::shared::events_queue",
    "devilution::shared::player",
    "devilution::shared::game_ticks",
    "devilution::shared::game_saves",
    "devilution::shared::game_loads",
    "devilution::shared::monster_type_info",
    "devilution::shared::max_monster_level",
    "devilution::shared::max_walk_frames",
    "devilution::shared::max_attack_frames",
    "devilution::shared::max_weapon_dam",
    "devilution::shared::player_class_attrs",

    # Options
    "devilution::GameTicksPerStep",
    "devilution::StepMode",

    # Monsters

    "devilution::ActiveMonsterCount",
    "devilution::Monsters",
    "devilution::ActiveMonsters",
    "devilution::MonsterKillCounts",

    # Objects

    "devilution::Objects",
    "devilution::ActiveObjects",

    # Diablo

    "devilution::PauseMode",

    # Items

    "devilution::ActiveItemCount",
    "devilution::ActiveItems",
    "devilution::Items",
    "devilution::dItem",

    # Automap

    "devilution::AutomapView",
    "devilution::AutomapTypeTiles",

    # Gendung

    "devilution::dungeon",

    "devilution::currlevel",
    "devilution::setlvlnum",
    "devilution::dFlags",
    "devilution::dMonster",
    "devilution::dObject",
    "devilution::dPiece",
    "devilution::dSpecial",
    "devilution::SOLData",

    # Trigs

    "devilution::trigs",
    "devilution::numtrigs",
]

# Types which should be exported
DEVILUTIONX_TYPES = [
    "devilution::monster_resistance",
    "devilution::inv_item",
    "devilution::AutomapTile",
]

def generate(binary_path):
    module_path = "devilutionx.py"
    content, regenerate = dbg2numpy.generate_numpy_module(
        DEVILUTIONX_VARS, binary_path, module_path,
        types_names=DEVILUTIONX_TYPES)
    if regenerate:
        open(module_path, "w").writelines(content)
