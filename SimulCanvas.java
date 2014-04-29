import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GraphicsConfiguration;
import java.awt.GraphicsEnvironment;
import java.awt.image.VolatileImage;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import javax.swing.JComponent;

public class SimulCanvas extends JComponent implements Runnable
{
	/** This frame delay limits the FPS to 60 */
	public static final double		FRAME_DELAY			= 50.0 / 3.0;

	public static final int			TPATH				= 1;
	public static final int			TOBJ				= 2;
	public static final int			THUM				= 3;
	public static final int			TEND				= 4;

	public static final char		HUMCHAR				= 'S';
	public static final char		ENDCHAR				= 'E';
	public static final char		OBCHAR				= 'O';
	public static final char		BLKCHAR				= 'B';

	private static final long		serialVersionUID	= 162410073428239152L;

	private VolatileImage			image;

	private GraphicsConfiguration	gc;

	private int						initialWidth;
	private int						initialHeight;
	private int[][]					sceneMap;

	private long					cycleTime, startCycleTime;

	private AtomicLong				cycleCount			= new AtomicLong(0);
	private AtomicBoolean			start;

	private Dimension				minDim				= new Dimension(200, 200);

	public SimulCanvas(int width, int height, String results, AtomicBoolean start, int[][] map)
	{
		super();
		this.start = start;
		sceneMap = map;
		setBackground(Color.WHITE);
		// Set the graphics configuration
		// Figure out the width and height of this canvas
		gc =
			GraphicsEnvironment.getLocalGraphicsEnvironment().getDefaultScreenDevice()
				.getDefaultConfiguration();
		initialWidth = width;
		initialHeight = height;
		new Thread(this).start();
	}

	@Override
	public Dimension getPreferredSize()
	{
		return new Dimension(initialWidth, initialHeight);
	}

	@Override
	public Dimension getMinimumSize()
	{
		return minDim;
	}

	@Override
	public void paint(Graphics g)
	{
		long newCycle = cycleCount.get();
		stepSimulation(cycleCount.compareAndSet(newCycle, newCycle + 1), (Graphics2D) g);
	}

	@Override
	public void run()
	{
		while (true) {
			long newCycle = cycleCount.get();
			stepSimulation(cycleCount.compareAndSet(newCycle, newCycle + 1),
				(Graphics2D) getGraphics());
		}
	}

	private void stepSimulation(boolean first, Graphics2D g2d)
	{
		startCycleTime = System.currentTimeMillis();
		if (first && g2d != null) {
			int width = getWidth(), height = getHeight();
			// Get the image graphics
			if (image == null || image.validate(gc) == VolatileImage.IMAGE_INCOMPATIBLE
				|| image.getWidth() != width || image.getHeight() != height)
				image = gc.createCompatibleVolatileImage(width, height);
			// Draw on it
			Graphics2D ig = image.createGraphics();
			ig.setColor(Color.red);
			ig.fillOval(0, 0, width / 2, height / 2);
			ig.dispose();
			g2d.drawImage(image, 0, 0, null);
		}
		syncFramerate();
	}

	/** <code>private void synchFramerate()</code>
	 * <p>
	 * Syncs the frame rate so that it is no faster than 1/FRAME_DELAY.
	 * </p> */
	private void syncFramerate()
	{
		cycleTime = System.currentTimeMillis();
		long difference = Math.round(FRAME_DELAY - (cycleTime - startCycleTime));
		if (difference > 0) {
			try {
				Thread.sleep(difference);
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
	}

}
