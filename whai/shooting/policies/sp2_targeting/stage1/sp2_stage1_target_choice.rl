import rules
import game_utils
import bounded_arg

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

    # Randomize target count, composition, and placement.
    act choose_target_count(frm BInt<3, 6> target_count)

    frm count : BInt<3, 6>
    count = target_count
    frm i : BInt<0, 6>
    i = 0
    while i.value != count.value:
        act choose_target_type(frm BInt<0, 4> target_type)
        if target_type.value == 0:
            board.units.append(make_infernus_squad())
        else if target_type.value == 1:
            board.units.append(make_intercessor_squad())
        else if target_type.value == 2:
            board.units.append(make_terminator_squad())
        else if target_type.value == 3:
            board.units.append(make_bladeguard_veteran_squad())
        else:
            board.units.append(make_hellblaster_squad())

        board.units.back().owned_by_player1 = true

        if i == 0:
            # Ensure at least one target is usually in range.
            act choose_target_x_near(frm BInt<12, 20> target_x_near)
            act choose_target_y_near(frm BInt<8, 12> target_y_near)
            board.units.back().move_to(make_board_position(target_x_near.value, target_y_near.value))
        else:
            act choose_target_x_far(frm BInt<12, 28> target_x_far)
            act choose_target_y_far(frm BInt<8, 12> target_y_far)
            board.units.back().move_to(make_board_position(target_x_far.value, target_y_far.value))
        board.units.back().arrange()

        i = i + 1

    frm source_id : UnitID
    source_id = 0

    act select_target(frm UnitID target_id) {
        target_id.get() >= 1,
        target_id.get() < board.units.size(),
        board[target_id].owned_by_player1,
        board[target_id].models.size() > 0,
        board[source_id].distance(board[target_id]) <= 12.0
    }

    board.attack = attack(board, source_id, target_id, false, false)
    subaction*(board) board.attack

fun unit_total_wounds(Unit unit) -> Int:
    let sum = 0
    let i = 0
    while i != unit.models.size():
        sum = sum + unit.models[i].wounds_left()
        i = i + 1
    return sum

fun score(Game g, Int player_id) -> Float:
    if g.board.units.size() < 2:
        return 0.0
    let remaining = 0
    let i = 1
    while i != g.board.units.size():
        remaining = remaining + unit_total_wounds(g.board.units[i])
        i = i + 1
    if player_id == 0:
        return float(0 - remaining)
    return float(remaining)

fun get_current_player(Game g) -> Int:
    return default_get_current_player(g)

fun get_num_players() -> Int:
    return 2

fun max_game_lenght() -> Int:
    return 500

fun pretty_print(Game g):
    print_indented(g)
