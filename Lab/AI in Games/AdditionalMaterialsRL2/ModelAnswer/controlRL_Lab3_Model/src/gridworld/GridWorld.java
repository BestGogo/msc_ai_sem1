package gridworld;

import utils.Vector2d;

import java.util.ArrayList;

/**
 * Created by dperez on 22/08/15.
 */
public class GridWorld
{
    public int size;
    public int[][] rewards;

    //ACTIONS FOR THE GRID WORLD
    public static final Vector2d RIGHT = new Vector2d(1, 0);
    public static final Vector2d LEFT = new Vector2d(-1, 0);
    public static final Vector2d UP = new Vector2d(0, -1);
    public static final Vector2d DOWN = new Vector2d(0, 1);

    public ArrayList<Vector2d> actions;

    public GridWorld()
    {
        generateWorld(3); //default of 3. Why not?
        generateActions();
    }

    public GridWorld(int size)
    {
        generateWorld(size);
        generateActions();
    }


    public void generateActions()
    {
        actions = new ArrayList<>(4);
        actions.add(RIGHT);
        actions.add(LEFT);
        actions.add(UP);
        actions.add(DOWN);
    }


    public void generateWorld(int size)
    {
        this.size = size;
        this.rewards = new int[size][size];

        for(int i = 0; i < size; ++i)
            for(int j = 0; j < size; ++j)
                this.rewards[i][j] = -1;

        this.rewards[0][0] = 0;
        this.rewards[size-1][size-1] = 0;
    }

    public static int getActionIndex(Vector2d act)
    {
        if(act == RIGHT) return 0;
        if(act == LEFT) return 1;
        if(act == UP) return 2;
        if(act == DOWN) return 3;
        return -1;
    }


    public int getReward (int row, int col)
    {
        return this.rewards[row][col];
    }

    public boolean isTerminal(int row, int col)
    {
        return (this.rewards[row][col] == 0);
    }



}
