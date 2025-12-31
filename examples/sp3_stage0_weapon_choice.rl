import rules
import game_utils

const TARGET_MODELS = 5

@classes
act play() -> Game:
    frm board : Board
    board.current_player = false
    board.current_decision_maker = false
    board.starting_player = false

    # Prevent stratagem branches from adding extra decisions.
    board.command_points[0] = 0
    board.command_points[1] = 0
    board.players_faction[0] = Faction::strike_force_octavius
    board.players_faction[1] = Faction::strike_force_octavius

    # Shooter (player 0)
    board.units.append(make_infernus_squad())
    board.units[0].owned_by_player1 = false
    board.units[0].move_to(make_board_position(10, 10))
    board.units[0].arrange()
    board.units[0].can_shoot = true
    board.units[0].has_shoot = false

    # Target (player 1)
    board.units.append(make_infernus_squad())
    board.units[1].owned_by_player1 = true
    board.units[1].move_to(make_board_position(12, 10))
    board.units[1].arrange()

    let source_id : UnitID
    source_id = 0
    let target_id : UnitID
    target_id = 1

    board.attack = attack(board, source_id, target_id, false, false)
    subaction*(board) board.attack

fun score(Game g, Int player_id) -> Float:
    if g.board.units.size() < 2:
        return 0.0
    let remaining = g.board.units[1].models.size()
    if player_id == 0:
        return float(TARGET_MODELS - remaining)
    return float(remaining)

fun get_current_player(Game g) -> Int:
    return default_get_current_player(g)

fun get_num_players() -> Int:
    return 2

fun max_game_lenght() -> Int:
    return 500

fun pretty_print(Game g):
    print_indented(g)
