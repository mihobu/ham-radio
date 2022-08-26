import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import datetime as dt
import maidenhead as mh
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Polygon, Rectangle
from string import ascii_uppercase

# Convert LON,LAT to X,Y coords in the plot space
def userland_to_cartesian(x, y, ax):
    x_cart, y_cart = ccrs.PlateCarree().transform_point(x, y, ax.projection)
    return x_cart, y_cart

# Covert X,Y coordinates in the plot space to LON,LAT
def cartesian_to_userland(x_cart, y_cart, ax):
    x, y = ax.projection.transform_point(x_cart, y_cart, ccrs.PlateCarree())
    return x, y

def draw_map_rect(lllat, lllon, urlat, urlon, ax, fc='#CC0000', ec=None, a=0.25, lw=0):
    L = 10
    
    lats1 = np.linspace(lllat, urlat, L)
    lons1 = np.ones(L) * lllon
    
    lats2 = np.ones(L) * urlat
    lons2 = np.linspace(lllon, urlon, L)

    lats3 = np.flip(np.linspace(lllat, urlat, L))
    lons3 = np.ones(L) * urlon

    lats4 = np.ones(L) * lllat
    lons4 = np.flip(np.linspace(lllon, urlon, L))

    lats = np.concatenate((lats1,lats2,lats3,lats4))
    lons = np.concatenate((lons1,lons2,lons3,lons4))
    
    XX = [cartesian_to_userland(xx, yy, ax) for xx, yy in np.stack([lons, lats], axis=-1)]
    return Polygon(XX, edgecolor=ec, facecolor=fc, alpha=a, linewidth=lw)

class MyGridMap:
    
    # Z-order values (zorder=):
    #  100 : shaded grid squares (worked)
    #  120 : grid square lines
    #  140 : grid field lines
    #  200 : grid field labels
    #  300 : date and time
    #  400 : City names
    
    # =================================================================
    # =================================================================
    def __init__(self, gridloc='EN80'):
        self.gridloc = gridloc
        self.fig = plt.figure(figsize=(12,6), dpi=200)
        self.fig.tight_layout(pad=0)
        self.cy, self.cx = mh.to_location(gridloc, center=True)
        #self.proj = ccrs.AzimuthalEquidistant(central_longitude=self.cx, central_latitude=self.cy)
        self.proj = ccrs.Stereographic(central_longitude=self.cx, central_latitude=self.cy)
        self.ax = self.fig.add_subplot(1, 1, 1, projection=self.proj)
        self.ax.set_global()

        # Uncomment these lines if you want to omit the border
        self.ax.axes.get_xaxis().set_visible(False)
        self.ax.axes.get_yaxis().set_visible(False)
        self.ax.set_frame_on(False)

        # Set map time
        self.maptime = dt.datetime.now(dt.timezone.utc)

    # =================================================================
    # =================================================================
    def DrawGridSquare(self, locator, fc='#339933', label=False): #, m=None, fc='#CC0000', ec=None, a=0.25, lw=0):
        assert len(locator) == 4, 'Grid square locator must be 4 characters long'

        lllat, lllon = mh.to_location(locator, center=False) # lower left lat/lon
        urlat = lllat + 1
        urlon = lllon + 2

        ec = '#000000'
        #fc = '#339933'
        a  = 0.5
        lw = 0.

        font = {
            'family': 'Menlo',
            'color':  '#333333',
            'weight': 'medium',
            'size': 5,
            'alpha': 1.0
        }
        
        poly = Rectangle(
            xy=[lllon, lllat], width=2, height=1,
            edgecolor=ec, facecolor=fc, alpha=a, linewidth=0, zorder=100,
            transform=ccrs.Geodetic()
        )
        self.ax.add_patch(poly)

        # Add label
        if label:
            cy, cx = mh.to_location(locator, center=True)
            plt.text(
                cx, cy, locator,
                ha='center', va='center',
                #color='black', fontsize=4,
                fontdict=font,
                transform=ccrs.Geodetic()
            )

    # =================================================================
    # =================================================================
    def AddFeatures(self):
        print('Adding features...', end='')

        # Add Ocean
        self.ax.add_feature(
            cfeature.NaturalEarthFeature(category='physical',scale='50m',name='ocean'),
            facecolor='#97B6E1',
            edgecolor='#000000',
            linewidth=0
        )

        # Add Land
        self.ax.add_feature(
            cfeature.NaturalEarthFeature(category='physical',scale='50m',name='land'),
            facecolor='#E3D796',
            edgecolor='#666600',
            linewidth=0.5
        )

        # Add States
        self.ax.add_feature(
            cfeature.NaturalEarthFeature(category='cultural',scale='50m',name='admin_1_states_provinces_lines'),
            facecolor='none',
            edgecolor='#AAAACC',
            linewidth=0.5
        )

        # Add Lakes
        self.ax.add_feature(
            cfeature.NaturalEarthFeature(category='physical',scale='50m',name='lakes'),
            facecolor='#97B6E1',
            edgecolor='#807C6E',
            linewidth=0.5
        )

        # Add Boundary Lines
        self.ax.add_feature(
            cfeature.NaturalEarthFeature(category='cultural',scale='50m',name='admin_0_boundary_lines_land'),
            facecolor='none',
            edgecolor='#3333CC',
            linewidth=0.5,
            linestyle='-.'
        )

        # Grid lines (2x1)
        self.ax.gridlines(
            draw_labels=False,
            xlocs=np.arange(-180,180,2),ylocs=np.arange(-90,90,1),
            color='#CCCCCC', lw=0.5, alpha=0.5,
            zorder=120
        )
        
        # Grid lines (20x10)
        self.ax.gridlines(
            draw_labels=False,
            xlocs=np.arange(-180,180,20),ylocs=np.arange(-90,90,10),
            color='#CCCCCC', lw=1.0, alpha=0.5,
            zorder=140
        )
        print('Done.')

    # =================================================================
    # =================================================================
    def SetExtent(self, wdegr, aspect):
        """
        Parameters
        ----------
        wdegr     : Minimum horizontal extent, in degrees
        aspect    : Aspect ratio
        ax        : Axis to set
        """

        self.wdegr = wdegr
        self.hdegr = wdegr / aspect

        # Can't exceed -90 (south pole) or +90 (north pole)
        lly = np.sign(self.cy-self.hdegr) * min(abs(self.cy-self.hdegr), 90)
        ury = np.sign(self.cy+self.hdegr) * min(abs(self.cy+self.hdegr), 90)

        # But longitude can overflow
        llx = self.cx - self.wdegr
        urx = self.cx + self.wdegr

        self.ax.set_extent([llx, urx, lly, ury])
    
    # =================================================================
    # =================================================================
    def CropMap(self, CROPFACTOR=0.01, ASPECT=0.75):
        minxu, maxxu = self.ax.get_xlim() # Get min and max X in userland space (plot coords)
        minyu, maxyu = self.ax.get_ylim() # Get min and max X in userland space (plot coords)
        Dx, Dy = maxxu - minxu, maxyu - minyu # Absolute width and height
        Cx, Cy = (minxu+maxxu)/2, (minyu+maxyu)/2 # Center
        newminxu, newmaxxu = Cx - Dx * CROPFACTOR, Cx + Dx * CROPFACTOR
        newminyu, newmaxyu = Cy - Dy * CROPFACTOR * ASPECT, Cy + Dy * CROPFACTOR * ASPECT
        self.ax.set_xlim(newminxu, newmaxxu)
        self.ax.set_ylim(newminyu, newmaxyu)

    # =================================================================
    # =================================================================
    def AddGridFieldLabels(self):
        print('Adding grid field labels...', end='')
        font = {
            'family': 'DIN Alternate',
            'color':  '#CC3366',
            'weight': 'bold',
            'size': 32,
            'alpha': 0.5
        }
        for gfa in ascii_uppercase[0:18]:
            print('.', end='')
            for gfb in ascii_uppercase[0:18]:
                gfloc = '{}{}'.format(gfa, gfb)
                gflat, gflon = mh.to_location(gfloc, center=True)
                plt.text(
                    gflon, gflat, gfloc,
                    ha='center', va='center',
                    transform=ccrs.PlateCarree(),
                    fontdict=font,
                    zorder=200
                ).set_clip_on(True)
        print('Done.')

    # =================================================================
    # =================================================================
    def AddGridSquareLabels(self, qso_df=None, newsquares=[]):
        print('Adding grid square labels...', end='')
        font = {
            'family': 'Menlo',
            'color':  '#333333',
            'weight': 'medium',
            'size': 5,
            'alpha': 1.0
        }

        # Get min and max X and Y coordinates in userland space
        minxu, maxxu = self.ax.get_xlim()
        minyu, maxyu = self.ax.get_ylim()

        # Get cartesian coordinates of the four corners
        lllon, lllat = userland_to_cartesian(minxu, minyu, self.ax)
        ullon, ullat = userland_to_cartesian(minxu, maxyu, self.ax)
        urlon, urlat = userland_to_cartesian(maxxu, maxyu, self.ax)
        lrlon, lrlat = userland_to_cartesian(maxxu, minyu, self.ax)
        print('UL {:7.2f} | {:7.2f} UR'.format(ullon, urlon))
        print('LL {:7.2f} | {:7.2f} LR'.format(lllon, lrlon))

        # Set the min and max latitudes for grid labels
        latrange = np.arange(np.floor(lllat/10)*10, np.ceil(urlat/10)*10+1, 10)

        # Pad the longitude extremes at higher latitudes
        maxabslat = max(abs(lllat), abs(urlat))
        #pad = int((maxabslat/90.)*40.) # Compute a pad of up to 40 extra degrees
        pad = 0.
        #print('pad={}'.format(pad))
        
        # Get longitude range, taking meridian crossing into account
        if (ullon>0)and(lllon>0)and(urlon<0)and(lrlon<0):
            # Case A: left edge < +180, right edge > -180 (i.e. map straddles 180th mer.)
            print('caseA')
            lo = np.floor(min(lllon,ullon)/20)*20
            hi = np.ceil(max(lrlon,urlon)/20)*20
            lonrange = np.concatenate([np.arange(lo, 180, 20),np.arange(-180, hi, 20)])
        elif (min(lllon,ullon)<0)and(max(lllon,ullon)>0)and(urlon<0)and(lrlon<0):
            # Case B: left edge crosses 180th mer.
            print('caseB')
            lo = np.floor(max(lllon,ullon)/20)*20
            hi = np.ceil(max(lrlon,urlon)/20)*20
            lonrange = np.concatenate([np.arange(lo, 180, 20),np.arange(-180, hi, 20)])
        elif (min(lrlon,urlon)<0)and(max(lrlon,urlon)>0)and(ullon>0)and(lllon>0):
            # Case C: right edge crosses 180th mer.
            print('caseC')
            lo = np.floor(min(lllon,ullon)/20)*20
            hi = np.ceil(min(lrlon,urlon)/20)*20
            lonrange = np.concatenate([np.arange(lo, 180, 20),np.arange(-180, hi, 20)])
        else:
            print('default case')
            # Default case: map does not cross 180th meridian
            lonrange = np.arange(np.floor(min(lllon,ullon)/20)*20, np.ceil(max(lrlon,urlon)/20)*20, 20)

        # Iterate
        for lon in lonrange:
            #print('{:.4f}'.format(lon), end='')
            for lat in latrange:
                gfloc = mh.to_maiden(lat, lon, precision=1)
                #print('{},{} -> {}'.format(lon, lat, gfloc))
                for gsrow in range(10):
                    for gscol in range(10):
                        gsloc = '{}{}{}'.format(gfloc, gsrow, gscol)
                        
                        # Fill in the worked squares, if requested
                        if qso_df is not None and (np.count_nonzero(qso_df['grid']==gsloc) > 0):
                            # Fill this grid square if worked
                            fillcolor = '#333399' if gsloc in newsquares else '#339933'
                            self.DrawGridSquare(gsloc, fc=fillcolor)
                            
                        # Label this grid square
                        cy, cx = mh.to_location(gsloc, center=True)
                        plt.text(
                            cx, cy, '{}{}'.format(gsrow, gscol),
                            ha='center', va='center',
                            #color='black', fontsize=4,
                            fontdict=font,
                            transform=ccrs.Geodetic()
                        ).set_clip_on(True)

        print('Done.')

    # =================================================================
    # =================================================================
    def AddDateTime(self):
        print('Adding date and time...', end='')
        # ADD DATE AND TIME
        ts = self.maptime.strftime("W8MHB grid square map centered on {} (%Y-%m-%d %H:%M UTC)".format(self.gridloc))
        self.ax.annotate(
            ts, xy=(0, 0), xycoords='axes fraction', fontsize=4, color='#6666CC',
            backgroundcolor='white',
            xytext=(5, 5), textcoords='offset points',
            ha='left', va='bottom', zorder=300)
        print('Done.')
    
    # =================================================================
    # =================================================================
    def SaveFig(self, abbr=None, fn=None):
        if abbr is None:
            abbr = self.gridloc
        if fn is None:
            fn = self.maptime.strftime('{}-%Y%m%d-%H%M%S.png'.format(abbr))
        opath = '/Users/mhb/OneDrive/hamradio/maps/Grid Maps/{}'.format(fn)
        print('Saving {}...'.format(fn), end='')
        self.fig.savefig(opath, bbox_inches='tight', pad_inches=0)
        print('Done.')
        
    # =================================================================
    # =================================================================
    def AddCities(self):
        font = {
            'family': 'Helvetica Neue',
            'color':  '#000000',
            'weight': 'medium',
            'size': 7,
            'alpha': 1.0
        }
        shpfilename = shpreader.natural_earth(resolution='10m', category='cultural', name='populated_places')
        reader = shpreader.Reader(shpfilename)
        places = reader.records()
        for place in places:
            #is_world_capital = (place.attributes['ADM0CAP'] == 1)
            if place.attributes['MEGACITY'] == 1:
                place_name = place.attributes['NAME']
                place_lat = place.attributes['LATITUDE']
                place_lon = place.attributes['LONGITUDE']
                self.ax.plot(
                    [place_lon], [place_lat],
                    color='black', marker='.', mew=0, ms=3, alpha=1.0,
                    transform=ccrs.Geodetic(),
                    zorder=400
                )
                self.ax.text(
                    place_lon+0.2, place_lat, place_name,
                    ha='left', va='center',
                    fontdict=font,
                    transform=ccrs.Geodetic(),
                    zorder=400
                ).set_clip_on(True)

    # =================================================================
    # =================================================================
    def AddCountries(self):
        shpfilename = shpreader.natural_earth(resolution='10m', category='cultural', name='admin_0_countries')
        reader = shpreader.Reader(shpfilename)
        places = reader.records()
        for place in places:
            place_name = place.attributes['NAME']
            place_lat = place.attributes['LABEL_Y']
            place_lon = place.attributes['LABEL_X']
            cond = place.attributes['LABELRANK']
            if cond <= 5:
                self.ax.text(
                    place_lon, place_lat, place_name,
                    ha='center', va='center',
                    fontdict={
                        'family': 'Helvetica Neue LT Std',
                        'color':  '#000000',
                        'stretch': 'condensed',
                        'weight': 'medium',
                        'size': 12,
                        'alpha': 0.7
                    },
                    transform=ccrs.Geodetic(),
                    zorder=400
                ).set_clip_on(True)