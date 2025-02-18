#!/usr/bin/env python3
"""
dbg2ctypes.py

A utility and module that generates a Python `ctypes` module from C
static variables and type information extracted from a binary with
embedded debug symbols.

This tool uses GDB with Python scripting to:

- Resolve static addresses and types of specified global/static
  variables.

- Traverse and convert complex C type definitions (structs, enums,
  typedefs, arrays, etc.)  into equivalent Python `ctypes`
  declarations.

- Automatically order type definitions to satisfy interdependencies
  and avoid forward reference issues.

Inputs:
  - Path to a binary with DWARF debug info.
  - A list of static/global variable names to be accessed from Python.

Output:
  - A standalone Python module that defines all required `ctypes`
    types and structures, provides named access to the specified
    variables via `ctypes` mapped to their static addresses.

Example use case:
  - Interact with memory-mapped shared objects in-place.
  - Inspect and manipulate static C structures from a live process
    (e.g., devilutionX-AI project).

Requirements:
  - GDB with Python scripting support.
  - Python 3.x

Note: this tool assumes that the binary is not stripped and includes
  sufficient DWARF debug information.

Author: Roman Penyaev <r.peniaev@gmail.com>
"""

try:
    import gdb
except:
    # Not from GDB context module does not exist
    pass

from graphlib import TopologicalSorter
import ctypes
import hashlib
import os
import pprint
import re
import sys
import traceback

def type_code_to_str(code):
    typecode_to_str = {
        gdb.TYPE_CODE_PTR:     'pointer',
        gdb.TYPE_CODE_ARRAY:   'array',
        gdb.TYPE_CODE_STRUCT:  'struct',
        gdb.TYPE_CODE_INT:     'integer',
        gdb.TYPE_CODE_CHAR:    'integer',
        gdb.TYPE_CODE_BOOL:    'integer',
        gdb.TYPE_CODE_ENUM:    'enum',
        gdb.TYPE_CODE_TYPEDEF: 'typedef',
    }
    if code in typecode_to_str:
        return typecode_to_str[code]
    return 'unknown'

def get_type_info(base_type, nesting):
    # Extract target type
    target_type = base_type
    try:
        target_type = base_type.target()
    except:
        pass

    # Represents the real type, after removing all layers of typedefs.
    target_type = target_type.strip_typedefs()

    type_info = {
        'base_type': base_type,
        'name': str(base_type),
        'sizeof': base_type.sizeof,
        'class': type_code_to_str(base_type.code),
    }

    if str(base_type) != str(target_type):
        type_info |= {
            'target_type': target_type,
            'target_name': str(target_type),
            'target_sizeof': target_type.sizeof,
            'target_class': type_code_to_str(target_type.code),
        }
    if base_type.code == gdb.TYPE_CODE_ENUM or \
       base_type.code == gdb.TYPE_CODE_INT or \
       base_type.code == gdb.TYPE_CODE_BOOL:
        type_info['is_signed'] = base_type.is_signed
    elif target_type.code == gdb.TYPE_CODE_ENUM or \
         target_type.code == gdb.TYPE_CODE_INT or \
         target_type.code == gdb.TYPE_CODE_BOOL:
        type_info['is_signed'] = target_type.is_signed

    if base_type.code == gdb.TYPE_CODE_ARRAY:
        afields = base_type.fields()
        assert len(afields) == 1
        atype = afields[0].type
        assert atype.code == gdb.TYPE_CODE_RANGE
        low, high = atype.range()
        num_elems = high + 1
        type_info['num_elems'] = num_elems
    elif base_type.code == gdb.TYPE_CODE_STRUCT and nesting == 0:
        fields = []
        # Iterate over the struct members
        for field in base_type.fields():
            offsetof = 0
            try:
                offsetof = field.bitpos // 8
            except:
                # Ignore static fields
                continue
            field_info = {
                'name': field.name,
                'offsetof': offsetof,
                'type': get_type_info(field.type, nesting + 1)
            }
            fields.append(field_info)
        type_info['fields'] = fields
    elif base_type.code == gdb.TYPE_CODE_ENUM and nesting == 0:
        values = []
        for value in base_type.fields():
            values.append({
                'name': value.name,
                'value': value.enumval
            })
        type_info['values'] = values

    return type_info

def lookup_dependent_types(type_infos_dict):
    def type_may_depend(type_info):
        # Fields of a struct, typedef or array are candidates for
        # further traversal
        return type_info['class'] in ('struct', 'typedef', 'array')

    def get_dependent_types(type_info, graph):
        if 'fields' in type_info:
            types = [f['type']['base_type'] for f in type_info['fields']]
        else:
            types = [type_info['target_type']]
        graph.add(type_info['name'], *[str(t) for t in types])

        return types

    def get_new_type_infos(type_infos_dict, types_to_resolve):
        new_type_infos = []
        for t in types_to_resolve:
            name = str(t)
            if name not in type_infos_dict:
                ti = get_type_info(t, 0)
                type_infos_dict[name] = ti
                new_type_infos.append(ti)
        return new_type_infos

    graph = TopologicalSorter()
    types_to_resolve = [t for ti in type_infos_dict.values() if type_may_depend(ti) \
                        for t in get_dependent_types(ti, graph)]
    while types_to_resolve:
        new_type_infos = get_new_type_infos(type_infos_dict, types_to_resolve)
        new_types_to_resolve = [t for ti in new_type_infos if type_may_depend(ti) \
                                for t in get_dependent_types(ti, graph)]

        if False:
            print(f"type_infos_dict: {type_infos_dict.keys()}")
            print(f"types_to_resolve: {[str(t) for t in types_to_resolve]}")
            print("-----------------------------------")

        types_to_resolve = new_types_to_resolve

    # Do topological sort to ensure correct types dependency
    sorted_type_names = graph.static_order()

    return sorted_type_names, type_infos_dict

def lookup_variables_names(names):
    seen = set()
    # Ensure names unique while preserving the order
    names = [n.strip() for n in names if not (n in seen or seen.add(n))]
    type_infos_dict = {}
    variables = []
    # Prevent including `<symbol name>` for an address
    gdb.execute('set print symbol off')
    for var_name in names:
        v = gdb.parse_and_eval(var_name)
        type_name = str(v.type)
        variables.append({
            'name': var_name,
            'addr': int(v.address),
            'type_name': type_name
        })
        if type_name not in type_infos_dict:
            type_infos_dict[type_name] = get_type_info(v.type, 0)

    sorted_type_names, type_infos_dict = lookup_dependent_types(type_infos_dict)

    if False:
        import json, sys
        class MyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, gdb.Type):
                    return str(obj)
                return super().default(obj)
        type_infos = [type_infos_dict[n] for n in sorted_type_names]
        json.dump(type_infos, sys.stderr, indent=2, cls=MyEncoder)
        print(f"\ntype_infos_len: {len(type_infos)}")
    if False:
        import json, sys
        json.dump(variables, sys.stderr, indent=2)
        print(f"\nvariables_len: {len(variables)}")

    return sorted_type_names, type_infos_dict, variables

def lookup_types_names(names):
    seen = set()
    # Ensure names unique while preserving the order
    names = [n.strip() for n in names if not (n in seen or seen.add(n))]
    type_infos_dict = {}
    for type_name in names:
        t = gdb.lookup_type(type_name)
        type_name = str(t)
        if type_name not in type_infos_dict:
            type_infos_dict[type_name] = get_type_info(t, 0)

    sorted_type_names, type_infos_dict = lookup_dependent_types(type_infos_dict)

    if False:
        import json, sys
        class MyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, gdb.Type):
                    return str(obj)
                return super().default(obj)
        type_infos = [type_infos_dict[n] for n in sorted_type_names]
        json.dump(type_infos, sys.stderr, indent=2, cls=MyEncoder)
        print(f"\ntype_infos_len: {len(type_infos)}")

    return sorted_type_names, type_infos_dict

def type_name_to_ctypes_primitive(typename):
    # Primitive type mapping
    ctypes_primitives = {
        'int':            'ctypes.c_int32',
        'unsigned int':   'ctypes.c_uint32',
        'short':          'ctypes.c_int16',
        'unsigned short': 'ctypes.c_uint16',
        'long':           'ctypes.c_int64',
        'unsigned long':  'ctypes.c_uint64',
        'char':           'ctypes.c_int8',
        'unsigned char':  'ctypes.c_uint8',
        'bool':           'ctypes.c_bool',
    }
    # Remove const and signed
    typename = re.sub(r'(\s+|^)(const|signed)(\s+|$)', ' ', typename).strip()
    typename = ctypes_primitives.get(typename, None)
    return typename

def strip_namespaces(s):
    # Stip namespaces
    s = s.split("::")[-1]
    # Replace special symbols with underscore
    s = re.sub(r"(<|>|\[|\]|\s+)", "_", s)
    # Replace python keywords
    s = re.sub(r"^None$", "None_", s)
    return s

def generate_type(type_info, type_infos_dict):
    def of_type(type_info, t):
        if type_info['class'] == t:
            return True
        if 'target_class' in type_info and \
           type_info['target_class'] == t:
            return True
        return False

    def blacklisted(type_info):
        # Skip all CPP standard classes
        return type_info['name'].startswith("std::")

    def to_ctypes(type_info, type_infos_dict):
        comment = ""
        if blacklisted(type_info):
            # Some types can be blacklisted, so we don't expose them and just
            # specify the number of bytes.
            comment = f" # Opaque \"{type_info['name']}\""
            typename = f"ctypes.c_uint8 * {type_info['sizeof']}"
        elif of_type(type_info, 'pointer'):
            # This handles all pointers as 'integer', otherwise types
            # can't be represented as numpy arrays, and the following
            # error is raised: "Unknown PEP 3118 data type specifier
            # ..."
            # TODO: make smarter
            sz = ctypes.sizeof(ctypes.c_void_p)
            if sz == 8:
                typename = "ctypes.c_uint64"
            elif sz == 4:
                typename = "ctypes.c_uint32"
            else:
                assert 0, f"Unsupported arch with a pointer type of size '{sz}'"
        elif of_type(type_info, 'integer'):
            # This handles all primitives, which are either declared
            # directly by the 'integer' base type, or through the
            # target type, like 'typedef', 'enum' or 'array'.
            if type_info['class'] == 'integer':
                typename = type_info['name']
            else:
                typename = type_info['target_name']
            typename = type_name_to_ctypes_primitive(typename)
            if 'num_elems' in type_info:
                assert type_info['class'] == 'array'
                typename += f" * {type_info['num_elems']}"
            elif type_info['class'] == 'enum':
                # Enum as integer, but use a comment to emphasize
                comment = f" # enum {type_info['name']}"
        elif type_info['class'] == 'typedef':
            # TODO: only struct is supported for now
            assert type_info['target_class'] == 'struct', "Support only 'typedef' of a 'struct'"
            typename = strip_namespaces(type_info['target_name'])
        else:
            # This handles the rest, e.g. 'struct'
            if 'num_elems' in type_info:
                assert type_info['class'] == 'array'
                if type_info['target_class'] == 'enum':
                    target_type_info = type_infos_dict[type_info['target_name']]
                    # This is the only place with recursion limited to
                    # one level of nesting to extract the target
                    # 'enum' type.
                    typename, comment = to_ctypes(target_type_info, type_infos_dict)
                elif type_info['target_class'] == 'struct':
                    typename = strip_namespaces(type_info['target_name'])
                elif type_info['target_class'] == 'array':
                    typename = strip_namespaces(type_info['target_name'])
                else:
                    # TODO: do not expect any other arrays
                    assert 0, "Support only 'array' of a 'struct', 'enum' or 'array'"
                typename += f" * {type_info['num_elems']}"
            else:
                typename = strip_namespaces(type_info['name'])

        return typename, comment

    out = []

    class_name = strip_namespaces(type_info['name'])

    if blacklisted(type_info):
        # Just skip blacklisted types
        pass
    elif type_info['class'] == 'enum':
        out.append(f"class {class_name}(enum.Enum):")
        values = type_info['values']
        assert(len(values))
        for val in values:
            val_name = strip_namespaces(val['name'])
            out.append(f"\t{val_name} = {val['value']}")
    elif type_info['class'] == 'struct':
        out.append(f"class {class_name}(ctypes.Structure):")
        fields = type_info['fields']
        assert(len(fields))

        out.append("\t_fields_ = [")
        for f in fields:
            field_name = strip_namespaces(f['name'])
            field_type, field_comment = to_ctypes(f['type'], type_infos_dict)
            out.append(f'\t\t("{field_name}", {field_type}),{field_comment}')
        out.append("\t]")
    elif type_info['class'] == 'typedef':
        typename, comment = to_ctypes(type_info, type_infos_dict)
        out.append(f"class {class_name}({typename}): pass")
    elif type_info['class'] == 'array':
        typename, comment = to_ctypes(type_info, type_infos_dict)
        out.append(f"class {class_name}({typename}): pass")
    elif type_info['class'] == 'integer':
        # Primitive types are replaced on ctypes alternatives,
        # see the `type_name_to_ctypes_primitive()`
        pass
    elif type_info['class'] == 'pointer':
        # Pointers just ignored, we don't expect variables with
        # pointer types
        pass
    else:
        assert 0, f"Unsupported type class '{type_info['class']}'"

    return out

def validate_generated_env(env, type_infos_dict):
    """Validates the created environment by comparing the `sizeof` the
    real, extracted structures and the `offsetof` of their fields with
    the generated structures."
    """
    # Copy of the type_infos dictionary, but with stripped type name. Sigh.
    env_type_infos_dict = {strip_namespaces(n): v for n, v in type_infos_dict.items()}

    for env_name, env_cls in env.items():
        if not isinstance(env_cls, type):
            continue
        if not hasattr(env_cls, "_fields_"):
            continue

        type_info = env_type_infos_dict[env_name]

        if type_info['class'] == 'typedef':
            # Get actual struct from typedef declaration
            assert type_info['target_class'] == 'struct'
            type_info = type_infos_dict[type_info['target_name']]

        cls_sizeof = ctypes.sizeof(env_cls)
        # For classes with pointers, the overall size of the class may
        # be naturally aligned, which is not the case for the ctypes
        # structure, where we make a pointer opaque. However, we are
        # lucky here because what actually matters is offsetof, which
        # should match.
        #assert type_info['sizeof'] == cls_sizeof
        for i, env_field in enumerate(env_cls._fields_):
            env_f_offset = getattr(env_cls, env_field[0]).offset
            env_f_sizeof = ctypes.sizeof(env_field[1])

            field = type_info['fields'][i]
            assert field['offsetof'] == env_f_offset
            # Why? See the comment above.
            #assert field['type']['sizeof'] == env_f_sizeof

def generate_types_and_variables(sorted_type_names, type_infos_dict, variables):
    content = []
    for n in sorted_type_names:
        ti = type_infos_dict[n]
        lines = generate_type(ti, type_infos_dict)
        if lines and content:
            content.append("")
        for line in lines:
            content.append(line)

    class Unquoted(str):
        def __repr__(self):
            return self

    def to_ctypes_or_strip(typename):
        name = type_name_to_ctypes_primitive(typename)
        if name:
            return name
        return strip_namespaces(typename)

    # Represent the class name without quotes so that it becomes a
    # real class instance in the `exec` block
    vars_list = [{**v, 'type': Unquoted(to_ctypes_or_strip(v['type_name']))} \
                 for v in variables]

    # Do pretty formatting for the variables list
    vars_lines = pprint.pformat(vars_list, sort_dicts=True, width=1).split("\n")
    vars_lines = ["       " + l for l in vars_lines]
    vars_lines[0] = "VARS = " + vars_lines[0][7:]

    generator_sha256 = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
    binary_path = gdb.current_progspace().filename
    binary_sha256 = hashlib.sha256(open(binary_path, "rb").read()).hexdigest()

    top_content = f"""
    #
    # WARNING:
    # WARNING: This file is auto-generated.
    # WARNING
    #
    # It was generated from the binary
    # {binary_path}
    # using debug information extracted via GDB's Python API.
    #
    # Do not edit this file manually. Any changes may be lost if the
    # file is regenerated. The structures and symbols defined here
    # reflect the state of the binary's debug info at the time of
    # generation. If the binary changes, this file must be
    # regenerated.
    #
    # At the bottom of this file, certain variables are defined
    # whose purpose is to control whether the content of this file
    # is no longer up-to-date and should be regenerated:
    #
    # 1. If the dbg2ctypes.py (actual generator) has been changed, the
    #    GENERATOR_SHA256 variable number won't match.
    #
    # 2. If the source binary has been changed, then the SHA256 of the
    #    binary will not match the BINARY_SHA256 variable.
    #
    # 3. If variable names do not match with VARS.
    #

    import ctypes
    import enum
    """
    bottom_content = f"""
    GENERATOR_SHA256 = '{generator_sha256}'
    BINARY_SHA256 = '{binary_sha256}'
    BINARY_PATH = '{binary_path}'
    """
    content = \
        [l.strip() for l in top_content.strip().split("\n")] + \
        [""] + content + [""] + \
        [l.strip() for l in bottom_content.strip().split("\n")] +\
        [""] + vars_lines

    def print_with_lines(content, f):
        for i, line in enumerate(content):
            print(f"{i+1}: {line}", file=f)

    try:
        # Validate
        compiled = compile("\n".join(content), "<string>", "exec")
        env = {}
        exec(compiled, env)
        validate_generated_env(env, type_infos_dict)

    except SyntaxError as e:
        print_with_lines(content, sys.stderr)
        print(file=sys.stderr)
        print(e, file=sys.stderr)
        return None
    except Exception as e:
        print_with_lines(content, sys.stderr)
        print(file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print(file=sys.stderr)
        print(e, file=sys.stderr)
        return None

    return content

def is_module_actual(variables_names, binary_path, module_path):
    try:
        with open(module_path, 'r', encoding='utf-8') as f:
            content = f.read()

        compiled = compile(content, "<string>", "exec")
        module = {}
        exec(compiled, module)

        generator_sha256 = hashlib.sha256(open(__file__, "rb").read()).hexdigest()
        if module["GENERATOR_SHA256"] != generator_sha256:
            return None
        if sorted([v['name'] for v in module["VARS"]]) != sorted(variables_names):
            return None
        binary_sha256 = hashlib.sha256(open(binary_path, "rb").read()).hexdigest()
        if module["BINARY_SHA256"] != binary_sha256:
            return None

        return content
    except:
        return None

def generate_ctypes_module(variables_names, binary_path, module_path=None):
    if (content := is_module_actual(variables_names, binary_path, module_path)):
        # Great, nothing to do. Regeneration is not required, thus False
        return content, False

    this_script = os.path.abspath(__file__)
    this_script_dir = os.path.dirname(this_script)
    this_module = os.path.splitext(os.path.basename(this_script))[0]

    script_for_gdb = f"""
    python import sys
    sys.path.insert(0, '{this_script_dir}')
    import {this_module}
    variables_names = {repr(variables_names)}
    sorted_type_names, type_infos_dict, variables = {this_module}.lookup_variables_names(variables_names)
    script = {this_module}.generate_types_and_variables(sorted_type_names, type_infos_dict, variables)
    print("\\n".join(script))
    """
    script_for_gdb = ";".join([l.strip() for l in script_for_gdb.strip().split("\n")])

    import subprocess
    try:
        res = subprocess.run([
            "gdb",
            "--batch",
            "--nx",
            "--eval-command", script_for_gdb,
            "--args", binary_path],
            check=True,
            capture_output=True)

        content = res.stdout.decode('utf-8')
        return content, True
    except subprocess.CalledProcessError as e:
        raise Exception(e.stderr.decode('utf-8'))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate ctypes bindings from debug info using GDB.")
    parser.add_argument("-b", "--binary", required=True,
                   help="Path to the source binary with debug information.")
    parser.add_argument("-o", "--output",
                        help="Output Python module file. If not specified, output goes to stdout.")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Force regeneration even if output exists or appears up-to-date.")
    parser.add_argument("variables", nargs="+",
                        help="One or more variable names to extract ctypes structure definitions for.")
    args = parser.parse_args()

    content, regenerate = generate_ctypes_module(
        args.variables, args.binary,
        module_path=None if args.force else args.output)

    if args.output:
        if regenerate:
            open(args.output, "w").writelines(content)
    else:
        print(content)
