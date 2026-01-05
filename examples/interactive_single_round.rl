import rules
import game_utils

# Single round demo: set factions, attach leaders, deploy, then play one round.
@classes
act play() -> Game:
    frm board : Board
    board.players_faction[0] = make_octavious_strike_force(board.reserve_units, false)
    board.players_faction[1] = make_octavious_strike_force(board.reserve_units, true)

    subaction*(board) attach_leaders = attach_leaders(board)
    subaction*(board) deploy = deploy(board)

    frm initial_models : Int[2]
    initial_models[0] = board.count_models(0)
    initial_models[1] = board.count_models(1)

    subaction*(board) round = round(board)

# Maximize opponent model losses over the round.
fun score(Game g, Int player_id) -> Float:
    return float(g.initial_models[1-player_id] - g.board.count_models(1-player_id)) / 10.0

fun get_current_player(Game g) -> Int:
    return default_get_current_player(g)

fun log_alive_models_player1(Game g) -> Int:
    return g.board.count_models(0)

fun log_alive_models_player2(Game g) -> Int:
    return g.board.count_models(1)

fun get_num_players() -> Int:
    return 2

fun max_game_lenght() -> Int:
    return 5000

fun pretty_print(Game game):
    print_indented(game)
