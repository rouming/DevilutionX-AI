# Bash completion for ./sprout.py and ./diablo-ai.py sprout
# Source this file (or add to ~/.bashrc): source /path/to/sprout-completion.bash

__sprout_meta() {
    local meta_file="$1" key="$2"
    python3 -c "
import json
key='$key'
try:
    m = json.load(open('$meta_file'))
    if key == 'heads':
        print('\n'.join(m.get('heads', {}).keys()))
    elif key == 'runs':
        print('\n'.join(m.get('runs', {}).keys()))
    elif key == 'groups':
        gs = sorted(set(r.get('group', '') for r in m.get('runs', {}).values()))
        print('\n'.join(g for g in gs if g))
except:
    pass
" 2>/dev/null
}

# Core sprout completion logic.
# Args: $1 = path to .metadata.json
#       $2 = index in words[] where sprout args begin
#            (1 for ./sprout.py, 2 for ./diablo-ai.py sprout)
# Expects: cur, prev, words, cword in parent scope.
__sprout_complete() {
    local meta_file="$1"
    local start="$2"

    local -a subcmds=(create clone persist remove edit rewind rename tree log show fetch)
    local -a value_flags=(
        --working --head --from-run --from-head --parent-run --parent-head
        --run --group --params --description --alias
    )

    # Complete value for the previous flag
    case "$prev" in
        --working)
            _filedir -d
            return ;;
        --head|--from-head|--parent-head)
            COMPREPLY=( $(compgen -W "$(__sprout_meta "$meta_file" heads)" -- "$cur") )
            return ;;
        --run|--from-run|--parent-run)
            COMPREPLY=( $(compgen -W "$(__sprout_meta "$meta_file" runs)" -- "$cur") )
            return ;;
        --group)
            COMPREPLY=( $(compgen -W "$(__sprout_meta "$meta_file" groups)" -- "$cur") )
            return ;;
        --params|--description|--alias)
            return ;;
    esac

    # Identify the active sprout subcommand
    local subcmd="" skip=0 i w
    for ((i = start; i < cword; i++)); do
        w="${words[$i]}"
        if [[ $skip -eq 1 ]]; then skip=0; continue; fi
        case "$w" in
            --working) skip=1 ;;
            --*) ;;
            *)
                for sc in "${subcmds[@]}"; do
                    [[ "$w" == "$sc" ]] && { subcmd="$w"; break 2; }
                done
                ;;
        esac
    done

    # No sprout subcommand yet: complete subcommand names or global flags
    if [[ -z "$subcmd" ]]; then
        if [[ "$cur" == -* ]]; then
            COMPREPLY=( $(compgen -W "--working --debug" -- "$cur") )
        else
            COMPREPLY=( $(compgen -W "${subcmds[*]}" -- "$cur") )
        fi
        return
    fi

    # Count positional args typed so far after the sprout subcommand
    local pos_count=0 in_sub=0 skip_next=0
    for ((i = start; i < cword; i++)); do
        w="${words[$i]}"
        if [[ $skip_next -eq 1 ]]; then skip_next=0; continue; fi
        if [[ $in_sub -eq 0 ]]; then
            [[ "$w" == "$subcmd" ]] && in_sub=1
            continue
        fi
        local is_value_flag=0
        for f in "${value_flags[@]}"; do
            [[ "$w" == "$f" ]] && { is_value_flag=1; break; }
        done
        if [[ $is_value_flag -eq 1 ]]; then
            skip_next=1
        elif [[ "$w" != --* ]]; then
            (( pos_count++ ))
        fi
    done

    # Complete flags for the current sprout subcommand
    if [[ "$cur" == -* ]]; then
        local flags="--debug"
        case "$subcmd" in
            create)  flags+=" --head --from-run --from-head --params --description --alias" ;;
            clone)   flags+=" --from-run --from-head --head --parent-run --parent-head --params --description --alias" ;;
            remove)  flags+=" --group --run --head --whole-branch" ;;
            edit)    flags+=" --run --head --params --description --alias" ;;
            rewind)  flags+=" --persist" ;;
            tree)    flags+=" --group --verbose" ;;
            log)     flags+=" --run --head" ;;
            show)    flags+=" --run --head --all" ;;
        esac
        COMPREPLY=( $(compgen -W "$flags" -- "$cur") )
        return
    fi

    # Complete positional arguments
    case "$subcmd" in
        create|clone)
            [[ $pos_count -eq 0 ]] && \
                COMPREPLY=( $(compgen -W "$(__sprout_meta "$meta_file" groups)" -- "$cur") )
            ;;
        persist|rewind)
            [[ $pos_count -eq 0 ]] && \
                COMPREPLY=( $(compgen -W "$(__sprout_meta "$meta_file" heads)" -- "$cur") )
            ;;
        rename)
            [[ $pos_count -eq 0 ]] && \
                COMPREPLY=( $(compgen -W "$(__sprout_meta "$meta_file" heads)" -- "$cur") )
            # pos_count==1 is the new name - no completion
            ;;
    esac
}

_sprout_py() {
    local cur prev words cword
    _init_completion || return

    # Locate --working value; fall back to "models"
    local working="models" i
    for ((i = 1; i < ${#words[@]}; i++)); do
        if [[ "${words[$i]}" == "--working" ]]; then
            (( i++ ))
            [[ $i -lt ${#words[@]} ]] && working="${words[$i]}"
            break
        fi
    done

    __sprout_complete "$working/.metadata.json" 1
}

_diablo_ai_py() {
    local cur prev words cword
    _init_completion || return

    local -a subcmds=(sprout play play-ai play-bot train-ai demos-il train-il list)

    # Completing the subcommand itself
    if [[ $cword -eq 1 ]]; then
        COMPREPLY=( $(compgen -W "${subcmds[*]}" -- "$cur") )
        return
    fi

    # Delegate to sprout completion when sprout is the active subcommand
    if [[ "${words[1]}" == "sprout" ]]; then
        # --working is hardcoded to "models" by diablo-ai.py
        __sprout_complete "models/.metadata.json" 2
        return
    fi
}

complete -F _sprout_py   sprout.py
complete -F _sprout_py   ./sprout.py
complete -F _diablo_ai_py diablo-ai.py
complete -F _diablo_ai_py ./diablo-ai.py
