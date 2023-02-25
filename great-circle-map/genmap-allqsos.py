#! /Users/mhb/anaconda3/envs/gridmaps2/bin/python

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import datetime as dt
import maidenhead as mh
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import sqlite3

# Covert X,Y coordinates in the plot space to LON,LAT
def cartesian_to_userland(x_cart, y_cart, ax):
    x, y = ax.projection.transform_point(x_cart, y_cart, ccrs.PlateCarree())
    return x, y

# ========================================================================
# Load QSO data directly from MacLoggerDX SQLite database
# ========================================================================

dbfile = '/Users/mhb/Documents/MLDX_Logs/MacLoggerDX.sql'
connstr = 'file:{}?mode=ro'.format(dbfile)
con = sqlite3.connect(connstr, uri=True)

# Get field names
rows = con.execute("PRAGMA table_info(qso_table_v007)").fetchall()
fields = [rec[1] for rec in rows]
#print(fields)

# Get log data
rows = con.execute("select * from qso_table_v007").fetchall()
con.close()

# Put log data into a data frame
qso_df = pd.DataFrame(rows, columns=fields)

# Keep only the 4-character grid square portion
qso_df['grid'] = qso_df.grid.astype(str).str.slice(0,4)

# Lower-casify the band_rx column
qso_df['band_rx'] = qso_df['band_rx'].str.lower()

# Generate a date string from the qso_start column
qso_df['qsodate'] = qso_df['qso_start'].apply(lambda x: dt.datetime.utcfromtimestamp(x).strftime('%Y%m%d'))

print('Loaded {} QSOs and {} distinct grid squares.'.format(
    len(qso_df),
    qso_df['grid'].nunique()
))

log_start = dt.datetime.strptime(qso_df['qsodate'].min(), "%Y%m%d").strftime("%Y-%m-%d")
log_end = dt.datetime.strptime(qso_df['qsodate'].max(), "%Y%m%d").strftime("%Y-%m-%d")

# ========================================================================
# Find list of NEW grid squares in the last N days
# ========================================================================

d = 7 # How many days?

cutoff = (dt.datetime.today() - dt.timedelta(days=d-1)).strftime('%Y%m%d')
qsos_old = qso_df[qso_df['qsodate']<cutoff]
qsos_new = qso_df[qso_df['qsodate']>=cutoff]
oldgrids = {}

for rec in qsos_old.iterrows():
    qsodate = rec[1]['qsodate']
    grid = rec[1]['grid']
    if grid not in oldgrids.keys():
        #print('Grid square {} was first worked on {}.'.format(grid, qsodate))
        oldgrids[grid] = qsodate

newgrids = {}
for rec in qsos_new.iterrows():
    qsodate = rec[1]['qsodate']
    grid = rec[1]['grid']
    if grid not in oldgrids.keys() and grid not in newgrids.keys():
        #print('Grid square {} was new this week ({}).'.format(grid, qsodate))
        newgrids[grid] = qsodate

newsquares = list(newgrids.keys())

print('Worked {} new grid squares in the last {} days (since {}):'.format(len(newsquares), d, cutoff))
print(np.sort(newsquares))

# Create a new data frame consisiting of only the unique list of grid squares
grid_df = pd.DataFrame(qso_df['grid'].unique(), columns=['grid'])

# Convert grid square location to latitude and longitude (center of grid square)
grid_df[['grlat', 'grlon']] = grid_df['grid'].apply(lambda g: mh.to_location(g, center=True)).tolist()

# Compute plot coords of my home location
lat_0, lon_0 = mh.to_location('EN80lb', center=True)

# Initialize figure
fig = plt.figure(figsize=(10,10), dpi=200)
fig.tight_layout(pad=0)
proj = ccrs.AzimuthalEquidistant(central_longitude=lon_0, central_latitude=lat_0)
ax = fig.add_subplot(1, 1, 1, projection=proj)

# Omit the border
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.set_frame_on(False)

# Compute dates - <d> days ago (from above) to today
new_start = (dt.datetime.today() - dt.timedelta(days=d-1)).strftime('%Y.%m.%d')
new_end = dt.datetime.today().strftime('%Y.%m.%d')

# Load the raster image
fname = "/Users/mhb/_Jupyter Notebooks/Grid Maps/eo_base_2020_clean_geo.png"
img = plt.imread(fname)
img = img[::-1]
ax.imshow(img, origin='lower', transform=ccrs.PlateCarree(), extent=[-180, 180, -90, 90])

# Add map features
#ax.add_feature(cfeature.OCEAN, color='#97B6E1')
#ax.add_feature(cfeature.LAND, facecolor='#FFF8DC', linewidth=0.5, edgecolor='#807C6E')
#ax.add_feature(cfeature.GSHHSFeature('intermediate', levels=[1], edgecolor='#666600', facecolor='#E3D796', linewidth=0.5))
#ax.background_img(name='Explorer', resolution='high')
#ax.add_feature(cfeature.LAKES, facecolor='#97B6E1', edgecolor='#807C6E', linewidth=0.5)
#ax.add_feature(cfeature.BORDERS, linestyle='-.', linewidth=0.5, edgecolor='#3333CC')

# Font setup
font = {
    'family': 'Menlo',
    'color':  '#333333',
    'weight': 'medium',
    'size': 5,
    'alpha': 1.0
}

# Compute map extremes
llx = grid_df.apply(lambda x: cartesian_to_userland(x.grlon, x.grlat, ax)[0], axis=1).min()
lly = grid_df.apply(lambda x: cartesian_to_userland(x.grlon, x.grlat, ax)[1], axis=1).min()
urx = grid_df.apply(lambda x: cartesian_to_userland(x.grlon, x.grlat, ax)[0], axis=1).max()
ury = grid_df.apply(lambda x: cartesian_to_userland(x.grlon, x.grlat, ax)[1], axis=1).max()

# Iterate over the records
for row in grid_df.iterrows():
    rec = row[1]
    
    # Plot line from HOME to grid square
    ax.plot(
        [lon_0, rec.grlon], [lat_0, rec.grlat],
        color='#3399FF80', marker='None', ls='-', lw=0.5, transform=ccrs.Geodetic(),
        zorder=10
    )

    # Plot grid squares (as dots)
    if rec.grid in newsquares:
        # New grid square this week
        ax.plot(
            [rec.grlon], [rec.grlat],
            color='#CC3300FF', marker='s', mec='#FF999980', mew=1, ms=5, transform=ccrs.Geodetic(),
            zorder=30
        )
    else:
        ax.plot(
            [rec.grlon], [rec.grlat],
            color='#0033CCFF', marker='s', mec='#6699FF80', mew=0.5, ms=1.5, transform=ccrs.Geodetic(),
            zorder=20
        )

# Prepare image description
message = 'ARS W8MHB • {} GRID SQUARES WORKED {} TO {}'.format(len(grid_df), log_start, log_end)
if len(newsquares) > 0:
    message = message + ' ({} NEW SINCE {})'.format(len(newsquares), new_start)

# Add a description in the lower left
ax.annotate(
    message, xy=(0., 0.), xycoords='axes fraction', color='#333333',
    #backgroundcolor='white',
    fontfamily='Helvetica Neue LT Std', fontstretch='condensed', fontweight='medium', fontsize=6.5,
    xytext=(2, 2), textcoords='offset points',
    ha='left', va='bottom', zorder=300)

p = 1000000
ax.set_xlim([llx-p, urx+p])
ax.set_ylim([lly-p, ury+p])

fn = '/Users/mhb/tmp/w8mhb-grid-square-map.jpg'
fig.savefig(fn, bbox_inches='tight', pad_inches=0)

os.system('/usr/local/bin/aws s3 cp /Users/mhb/tmp/w8mhb-grid-square-map.jpg s3://downloads.monkeywalk.com/ham-radio/w8mhb-grid-square-map.jpg')
