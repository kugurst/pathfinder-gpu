import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.GraphicsConfiguration;
import java.awt.GraphicsEnvironment;
import java.awt.Image;
import java.awt.Transparency;
import java.awt.image.BufferedImage;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;

import javax.swing.JComponent;

public class SimulCanvas extends JComponent implements Runnable
{
	/** This frame delay limits the FPS to 60 */
	public static final double    FRAME_DELAY      = 1.0 / 60.0 * 1000.0;
	public static final int       TARGET_FPS       = 60;
	public static int             MOVE_RATE        = TARGET_FPS;

	public static final int       TPATH            = 1;
	public static final int       TOBJ             = 2;
	public static final int       THUM             = 3;
	public static final int       TEND             = 4;
	public static final int       TSTART           = 5;

	public static final int       PATH_LEN         = 6;

	public static final char      HUMCHAR          = 'S';
	public static final char      ENDCHAR          = 'E';
	public static final char      OBCHAR           = 'O';
	public static final char      BLKCHAR          = 'B';

	private static final long     serialVersionUID = 162410073428239152L;

	private Image                 image;

	private GraphicsConfiguration gc;

	private int                   initialWidth;
	private int                   initialHeight;

	private int[][]               sceneMap, backingMap;
	private List<Human>           humans           = new LinkedList<Human>();
	private int[]                 perHumItr;

	private long                  cycleTime, startCycleTime;

	private AtomicLong            cycleCount       = new AtomicLong(0);
	private AtomicBoolean         start;

	private Dimension             minDim           = new Dimension(200, 200);

	public SimulCanvas(int width, int height, String results, AtomicBoolean start, int[][] map)
	{
		super();
		this.start = start;
		sceneMap = map;
		backingMap = new int[sceneMap.length][sceneMap[0].length];
		// go through each row
		for (int y = 0; y < sceneMap.length; y++)
			for (int x = 0; x < sceneMap[0].length; x++)
				backingMap[y][x] = sceneMap[y][x];
		setBackground(Color.WHITE);
		// Set the graphics configuration
		// Figure out the width and height of this canvas
		gc =
		    GraphicsEnvironment.getLocalGraphicsEnvironment().getDefaultScreenDevice()
		        .getDefaultConfiguration();
		initialWidth = width / 2;
		initialHeight = height / 2;
		// Build the paths
		buildPaths(results);
		// Make the human iterations array
		perHumItr = new int[humans.size()];
		for (int i = 0; i < perHumItr.length; i++)
			perHumItr[i] = 1;
		new Thread(this).start();
	}

	private void buildPaths(String results)
	{
		// Split the results into lines
		String[] lines = results.split("\n");
		// Parse lines in group of two
		for (int i = 0; i < lines.length; i += 2) {
			Human hum = new Human();
			// Get the name
			hum.name = lines[i].split("\\ ")[1];
			hum.name = hum.name.substring(0, hum.name.length() - 1);
			// Build the path
			String[] pathTok =
			    lines[i + 1].substring(PATH_LEN, lines[i + 1].length()).split("\\,\\ ");
			for (int j = 0; j < pathTok.length; j++) {
				// Parse the coordinates
				String coordS = pathTok[j].replaceAll("\\(", "").replaceAll("\\)", "");
				String[] coords = coordS.split("\\,");
				Integer[] coord = new Integer[2];
				coord[0] = Integer.parseInt(coords[0]);
				coord[1] = Integer.parseInt(coords[1]);
				// For the first element, set our position
				if (j == 0) {
					hum.posX = coord[0];
					hum.posY = coord[1];
				}
				// For the last element, set our goal
				else if (j == pathTok.length - 1) {
					hum.goalX = coord[0];
					hum.goalY = coord[1];
				}
				// Add the coordinate to the path
				hum.path.add(coord);
			}
			humans.add(hum);
		}
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
		stepSimulation((Graphics2D) g);
	}

	@Override
	public void run()
	{
		while (true)
			stepSimulation((Graphics2D) getGraphics());
	}

	private synchronized void stepSimulation(Graphics2D g2d)
	{
		startCycleTime = System.currentTimeMillis();
		if (g2d != null) {
			int width = getWidth(), height = getHeight();
			if (width == 0 || height == 0) return;
			// Get the image graphics
			if (image == null || image.getWidth(null) != width || image.getHeight(null) != height) // image.validate(gc)
			                                                                                       // ==
			                                                                                       // VolatileImage.IMAGE_INCOMPATIBLE
			    image = gc.createCompatibleImage(width, height, Transparency.OPAQUE); // gc.createCompatibleVolatileImage(width,
			// height);
			// Draw on it
			Graphics2D ig = ((BufferedImage) image).createGraphics();
			ig.setBackground(Color.white);
			ig.clearRect(0, 0, width, height);
			// Draw the map
			drawMap(ig, width, height);
			// Move the humans
			if (start.get()) {
				animateHumans(ig, width, height);
				moveHumans();
			}
			g2d.drawImage(image, 0, 0, null);
			ig.dispose();
			cycleCount.incrementAndGet();
		}
		syncFramerate();
	}

	private void animateHumans(Graphics2D ig, int width, int height)
	{
		// Calculate grid widths
		int mapWidth = sceneMap[0].length;
		int mapHeight = sceneMap.length;
		int gridWidth = (int) Math.floor((double) width / mapWidth);
		int gridHeight = (int) Math.floor((double) height / mapHeight);
		// Number of cycles left in this second
		int cycles = MOVE_RATE - (int) cycleCount.get() % MOVE_RATE;
		ig.setColor(Color.blue);

		for (int i = 0; i < humans.size(); i++) {
			Human hum = humans.get(i);
			// If the grid locations are -1, then we need to calculate its position
			if (hum.gridX == -1 && hum.gridY == -1) {
				hum.gridX = hum.posX * gridWidth;
				hum.gridY = hum.posY * gridHeight;
				continue;
			}
			// if (cycleCount.get() % MOVE_RATE == 0) System.out.println("name: " + hum.name
			// + ", gridX: " + hum.gridX + ", gridY: " + hum.gridY);
			// Figure out their screen position
			int destX = hum.posX * gridWidth;
			int destY = hum.posY * gridHeight;
			// if (cycleCount.get() % MOVE_RATE == 0) System.out.println("name: " + hum.name
			// + ", destX: " + destX + ", destY: " + destY);
			// Get the distance to animate them
			int diffX = destX - hum.gridX;
			int diffY = destY - hum.gridY;
			// if (cycleCount.get() % MOVE_RATE == 0) System.out.println("name: " + hum.name
			// + ", diffX: " + diffX + ", diffY: " + diffY);
			// Assign their new position
			hum.gridX += diffX / cycles;
			hum.gridY += diffY / cycles;
			if (cycles == 1) {
				hum.gridX = destX;
				hum.gridY = destY;
			}
			// Draw them
			ig.fillOval(hum.gridX, hum.gridY, gridWidth, gridHeight);
			// if (travX == 0 && travY == 0 && cycleCount.get() % MOVE_RATE == 0) System.out
			// .println("grid dims: (" + gridWidth + ", " + gridHeight + ")");
		}
	}

	private void moveHumans()
	{
		// Once every second
		if (cycleCount.get() % MOVE_RATE == 0) {
			int pos = 0;
			for (Human hum : humans) {
				int curItr = perHumItr[pos];
				// If it's home, don't move it
				if (curItr < hum.path.size()) {
					Integer[] next = hum.path.get(curItr);
					hum.posX = next[0];
					hum.posY = next[1];
					// Increase our iteration count
					perHumItr[pos]++;
				}
				// } else System.out.println("name: " + hum.name + ", pos: (" + hum.posX + ", "
				// + hum.posY + "), grid: (" + hum.gridX + ", " + hum.gridY + ")");
				pos++;
			}
		}
	}

	private void drawMap(Graphics2D ig, int width, int height)
	{
		// Calculate grid widths
		int mapWidth = sceneMap[0].length;
		int mapHeight = sceneMap.length;
		int gridWidth = (int) Math.floor((double) width / mapWidth);
		int gridHeight = (int) Math.floor((double) height / mapHeight);
		int lastGridWidth = width - gridWidth * (mapWidth - 1) - 1;
		int lastGridHeight = height - gridHeight * (mapHeight - 1) - 1;

		// Draw the grid
		for (int y = 0; y < mapHeight; y++) {
			int drawHeight = y == mapHeight - 1 ? lastGridHeight : gridHeight;
			for (int x = 0; x < mapWidth; x++) {
				int drawWidth = x == mapWidth - 1 ? lastGridWidth : gridWidth;
				int pointType = sceneMap[y][x];
				// Don't fill paths
				if (pointType == TPATH) {
					ig.setColor(Color.red);
					ig.clearRect(x * gridWidth, y * gridHeight, drawWidth, drawHeight);
				}
				// Color goals green
				else if (pointType == TEND) {
					ig.setColor(Color.green);
					ig.fillRect(x * gridWidth, y * gridHeight, drawWidth, drawHeight);
				}
				// Color start positions yellow
				else if (pointType == TSTART) {
					ig.setColor(Color.yellow);
					ig.fillRect(x * gridWidth, y * gridHeight, drawWidth, drawHeight);
				}
				// Color obstacles red
				else if (pointType == TOBJ) {
					ig.setColor(Color.black);
					ig.fillRect(x * gridWidth, y * gridHeight, drawWidth, drawHeight);
				}
			}
		}
	}

	/**
	 * <code>private void synchFramerate()</code>
	 * <p>
	 * Syncs the frame rate so that it is no faster than 1/FRAME_DELAY.
	 * </p>
	 */
	private void syncFramerate()
	{
		cycleTime = System.currentTimeMillis();
		long difference = Math.round(FRAME_DELAY - (cycleTime - startCycleTime));
		if (difference > 0) try {
			Thread.sleep(difference);
		} catch (InterruptedException e) {
			e.printStackTrace();
		}
	}

	class Human
	{
		String          name;
		int             posX, posY, goalX, goalY, gridX = -1, gridY = -1;
		// Blank their start location
		int             curPntType = TSTART;
		// A list of x,y pairs
		List<Integer[]> path       = new LinkedList<Integer[]>();

		@Override
		public int hashCode()
		{
			final int prime = 31;
			int result = 1;
			result = prime * result + getOuterType().hashCode();
			result = prime * result + (name == null ? 0 : name.hashCode());
			return result;
		}

		@Override
		public boolean equals(Object obj)
		{
			if (this == obj) return true;
			if (obj == null) return false;
			if (!(obj instanceof Human)) return false;
			Human other = (Human) obj;
			if (!getOuterType().equals(other.getOuterType())) return false;
			if (name == null) {
				if (other.name != null) return false;
			} else if (!name.equals(other.name)) return false;
			return true;
		}

		private SimulCanvas getOuterType()
		{
			return SimulCanvas.this;
		}
	}
}
