package gridworld;

/**
 * Created by dperez on 22/08/15.
 */
public class GridWorldCenter extends GridWorld
{
    public GridWorldCenter(int size)
    {
        super(size);
    }

    public void generateWorld(int size)
    {
        this.size = size;
        this.rewards = new int[size][size];

        for(int i = 0; i < size; ++i)
            for(int j = 0; j < size; ++j)
                this.rewards[i][j] = -1;

        this.rewards[size/2][size/2] = 0;
    }

}
