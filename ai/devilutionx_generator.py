"""
devilutionx_generate.py - Generator of the devilutionx.py module
"""
import dbg2ctypes

DEVILUTIONX_VARS = [
    "devilution::shared::input_queue",
    "devilution::shared::events_queue",
    "devilution::shared::player",
    "devilution::shared::game_ticks",
    "devilution::shared::game_saves",
    "devilution::shared::game_loads",

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
    "devilution::dItem",

    # Gendung

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

def generate(binary_path):
    module_path = "devilutionx.py"
    content, regenerate = dbg2ctypes.generate_ctypes_module(
        DEVILUTIONX_VARS, binary_path, module_path)
    if regenerate:
        open(module_path, "w").writelines(content)
