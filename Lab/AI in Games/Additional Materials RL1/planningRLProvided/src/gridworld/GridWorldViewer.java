package gridworld;

import algorithms.planning.PlanningAlgorithm;
import utils.JEasyFrame;
import utils.Pair;
import utils.Vector2d;

import javax.swing.*;
import javax.swing.text.Document;
import java.awt.*;
import java.awt.event.MouseEvent;
import java.awt.event.MouseListener;
import java.util.ArrayList;

/**
 * Created by dperez on 26/08/15.
 */
public class GridWorldViewer extends JComponent
{
    public GridWorld gw;
    public PlanningAlgorithm alg;
    public double[][] values;

    private Dimension size;
    public int cellSize = 30;

    private double min, max;
    private JEasyFrame frame;
    public Point pointMouseClicked;


    private int prevRow, prevCol;
    private Graphics2D g;


    public GridWorldViewer(GridWorld gw, PlanningAlgorithm alg, double[][] values) {
        this.gw = gw;
        this.values = values;
        this.alg = alg;
        pointMouseClicked = null;
        prevRow = prevCol = -1;

        size = new Dimension(values.length * cellSize, values[0].length * cellSize);

        frame = new JEasyFrame(this, "Gridworld Viewer");

        //frame.addMouseListener(new MouseListener() {
        this.addMouseListener(new MouseListener() {
            @Override
            public void mouseClicked(MouseEvent e) {
                pointMouseClicked = e.getPoint();
            }

            public void mousePressed(MouseEvent e) {
            }

            public void mouseEntered(MouseEvent e) {
            }

            public void mouseReleased(MouseEvent e) {
            }

            public void mouseExited(MouseEvent e) {
            }
        });


    }

    public void paintStateValueFunction(int row, int col, double val)
    {
        double normVal = normalise(val, min, max);
        int green = 255 - (int) (normVal*255);
        green = Math.max(0, Math.min(255,green));

        Color sqColor = new Color(70,green,255);

        g.setColor(Color.black);
        g.drawRect(row * cellSize, col * cellSize, cellSize, cellSize);
        g.setColor(sqColor);
        g.fillRect(row * cellSize, col * cellSize, cellSize, cellSize);
    }


    /**
     * Main method to paint the game
     * @param gx Graphics object.
     */
    public void paintComponent(Graphics gx)
    {
        g = (Graphics2D) gx;

        //For a better graphics, enable this: (be aware this could bring performance issues depending on your HW & OS).
        g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

        //g.setColor(Types.LIGHTGRAY);
        g.setColor(Color.black);
        g.fillRect(0, size.height, size.width, size.height);

        for(int i = 0; i < values.length; ++i)
            for(int j = 0; j < values[0].length; ++j)
            {
                paintStateValueFunction(i, j, values[i][j]);
            }


        if(pointMouseClicked != null)
        {
            g.setColor(Color.white);
            g.drawRect(prevCol * cellSize, prevRow * cellSize, cellSize, cellSize);
        }

    }

    /**
     * Paints the grid world.
     */
    public void paint(double min, double max, double values[][])
    {
        this.min = min;
        this.max = max;
        this.values = values;

        if(pointMouseClicked != null)
        {
            int row = (pointMouseClicked.y / cellSize);
            int col = pointMouseClicked.x / cellSize;

            if(row != prevRow || col != prevCol) {

                if (row >= 0 && row < values.length && col >= 0 && col < values[row].length) {

                    String valStr = String.format("Gridworld %dx%d [%d:%d] %.4f at (%d,%d)",
                            values.length, values[0].length, (int) this.min, (int) this.max, values[row][col], row, col);
                    frame.setTitle(valStr);

                    String policyStr = "[";
                    double policyProbs[] = new double[this.gw.actions.size()];
                    for (int i = 0; i < policyProbs.length; ++i) {
                        Vector2d act = this.gw.actions.get(i);
                        policyProbs[i] = alg.prob(row, col, act, this.gw.actions);
                        policyStr += policyProbs[i] + ",";
                    }
                    policyStr += "]";

                    System.out.println(policyStr + " " + alg.policy.getPolicyStr());

                }
            }

            prevRow = row;
            prevCol = col;

        } else {

            frame.setTitle("Gridworld " + values.length + "x" + values[0].length +
                    " [min: " + (int) this.min + ", max" + (int) this.max + "]");
        }


        this.repaint();
    }

    /**
     * Gets the dimensions of the window.
     * @return the dimensions of the window.
     */
    public Dimension getPreferredSize() {
        return size;
    }

    public double normalise(double a_value, double a_min, double a_max)
    {
        if(a_min < a_max)
            return (a_value - a_min)/(a_max - a_min);
        else    // if bounds are invalid, then return 0
            return 0;
    }

}
