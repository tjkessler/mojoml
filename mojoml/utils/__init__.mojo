from algorithm import Static2DTileUnitFunc


fn tile[tiled_fn: Static2DTileUnitFunc, tile_x: Int, tile_y: Int](
        end_x: Int, end_y: Int) -> None:
    """ Tiling helper function.
    """

    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)
