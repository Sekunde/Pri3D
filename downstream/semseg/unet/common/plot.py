import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from matplotlib.pyplot import *
from PIL import Image

colors = [ 'xkcd:blue',
           'xkcd:red',
           'xkcd:purple',
           'xkcd:orchid',
           'xkcd:orange',
           'xkcd:grey',
           'xkcd:teal',
           'xkcd:sienna',
           'xkcd:azure',
           'xkcd:green',
           'xkcd:black',
           'xkcd:goldenrod']

def plot_curves(curves, 
                xlabel='% Dataset Labeled\n(ScanNet-5-Recon)',
                xlim=[4, 36],
                xticks=np.arange(5, 35, 5),
                xticklabels=False,
                ylabel='mIoU', 
                ylim=[0.2, 0.65],
                yticks=np.arange(0.2, 0.65, 0.05),
                if_grid=True,
                image_name='test.png'):
    font = {'family' : 'Times New Roman',
            'size'   : 11}
    matplotlib.rc('font', **font)

    fig, subplot = plt.subplots(1,1)
    fig.set_size_inches(4.5, 4.0)
    subplot.set(xlim=xlim, ylim=ylim, xlabel=xlabel, ylabel=ylabel)
    subplot.set(xticks=xticks, yticks=yticks)
    if xticklabels:
        import matplotlib.ticker as mtick
        fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
        xticks = mtick.FormatStrFormatter(fmt)
        subplot.xaxis.set_major_formatter(xticks)

    subplot.grid(if_grid)

    for idx, curve in enumerate(curves):
        name = ''
        fmt=''
        marker = ''
        markersize = 7.25
        linewidth = 2.75
        color = colors[idx%len(colors)]


        if 'name' in curve:
            name = curve['name']
        if 'marker' in curve:
            marker = curve['marker']
        if 'markersize' in curve:
            marker_size = curve['markersize']
        if 'color' in curve:
            color = curve['color']
        if 'linewidth' in curve:
            linewidth = curve['linewidth']
        if 'fmt' in curve:
            fmt = curve['fmt']

        x = curve['x']
        y = curve['y']
        
        subplot.plot(x, y, fmt, label=name, marker=marker, markersize=markersize, linewidth=linewidth, color=color)

    subplot.legend(loc='best')
    fig.tight_layout()
    plt.show()
    fig.savefig(image_name, dpi=600)
    image = Image.open(image_name)
    w, h = image.size
    image.crop((75, 75, w - 75, h - 60)).save(image_name)


#--------------------------------------------------------------------------------------------------

#def scratch_resnet18():
#    '''
#    '''
#    data = [
#        {'name': 'ScratchView', 'x': [1, 5, 10, 20, 40, 60, 80, 100], 'y': [17.8(1), 26.3(5), 30.6(10), 35.5, 41.0, 43.4, 45.8, 47.7], 'marker': 'o'},
#        {'name': 'ScratchChunk', 'x': [1, 5, 10, 20, 40, 60, 80, 100], 'y': [17.7(1), 25.7(5), 28.3(10), 33.4, 39.4, 42.1, 44.6, 46.3], 'marker': 'o'},
#        {'name': 'ScratchCombine', 'x': [1, 5, 10, 20, 40, 60, 80, 100], 'y': [17.6(1), 26.8(5), 31.0(10), 35.4, 41.9, 44.2, 46.8, 48.2], 'marker': 'o'},
#        {'name': 'Scratch', 'x': [1, 5, 10, 20, 40, 60, 80, 100], 'y': [8.8(1), 11.4(5), 13.5(10), 18.3, 24.7, 30.2, 34.5, 37.5], 'marker': 'o'},
#    ]
#    plot_curves(curves=data,
#                xlabel='Percentage of Training Images',
#                ylabel='mIoU',
#                xlim=[10, 110],
#                ylim=[17.0, 50.0],
#                xticks=[20, 40, 60, 80, 100],
#                yticks=np.arange(17.0, 50.0, 5.0),
#                if_grid=True, 
#                image_name='data_efficient_scratch_resnet18.jpg')
#
#def scratch_resnet50():
#    '''
#    '''
#    data = [
#        {'name': 'ScratchView', 'x': [1, 5, 10, 20, 40, 60, 80, 100], 'y': [20.9(1), 29.8(5), 34.4(10), 38.5, 44.4, 47.2, 49.2, 50.6], 'marker': 'o'},
#        {'name': 'ScratchChunk', 'x': [1, 5, 10, 20, 40, 60, 80, 100], 'y': [18.8(1), 29.2(5), 32.0(10), 37.4, 42.7, 46.1, 48.2, 49.4], 'marker': 'o'},
#        {'name': 'ScratchCombine', 'x': [1, 5, 10, 20, 40, 60, 80, 100], 'y': [21.6(1), 31.5(5), 35.6(10), 40.4, 45.8, 48.7, 50.7, 51.8], 'marker': 'o'},
#        {'name': 'Scratch', 'x': [1, 5, 10, 20, 40, 60, 80, 100], 'y': [9.4(1), 13.0(5), 14.7(10), 19.4, 26.7, 32.2, 36.6, 39.1], 'marker': 'o'},
#    ]
#    plot_curves(curves=data,
#                xlabel='Percentage of Training Images',
#                ylabel='mIoU',
#                xlim=[10, 110],
#                ylim=[18.0, 53.0],
#                xticks=[20, 40, 60, 80, 100],
#                yticks=np.arange(18.0, 53.0, 5.0),
#                if_grid=True, 
#                image_name='data_efficient_scratch_resnet50.jpg')


def imagenet_resnet18():
    '''
    '''
    data = [
        {'name': 'Pri3D (View)', 'x': [20, 40, 60, 80, 100], 'y': [43.2, 48.7, 51.1, 52.9, 54.4], 'marker': 'o'},
        {'name': 'Pri3D (Geo)', 'x': [20, 40, 60, 80, 100], 'y': [43.9, 49.5, 51.8, 53.7, 55.3], 'marker': '^'},
        {'name': 'Pri3D (View+Geo)', 'x': [20, 40, 60, 80, 100], 'y': [45.1, 50.2, 52.2, 54.1, 55.7], 'marker': 's'},
        {'name': 'ImageNet (IN)', 'x': [20, 40, 60, 80, 100], 'y': [36.7, 43.9, 46.9, 49.8, 51.0], 'marker': 'X'},
        {'name': 'MoCoV2-supIN->SN', 'x': [20, 40, 60, 80, 100], 'y': [37.9, 44.9, 48.4, 49.8, 52.9], 'marker': 'D'},
    ]
    plot_curves(curves=data,
                xlabel='Percentage of Training Images',
                ylabel='mIoU',
                xlim=[15, 105],
                ylim=[36, 56],
                xticklabels=True,
                xticks=[20, 40, 60, 80, 100],
                yticks=np.arange(36, 56, 5),
                if_grid=True, 
                image_name='data_efficient_imagenet_resnet18.jpg')

def imagenet_resnet50():
    '''
    '''
    data = [
        {'name': 'Pri3D (View)', 'x': [20, 40, 60, 80, 100], 'y': [51.5, 56.0, 58.4, 59.4, 61.3], 'marker': 'o'},
        {'name': 'Pri3D (Geo)', 'x': [20, 40, 60, 80, 100], 'y': [50.9, 55.8, 58.2, 59.7, 61.1], 'marker': '^'},
        {'name': 'Pri3D (View + Geo)', 'x': [20, 40, 60, 80, 100], 'y': [52.1, 56.2, 58.9, 60.3, 61.7], 'marker': 's'},
        {'name': 'ImageNet (IN)', 'x': [20, 40, 60, 80, 100], 'y': [40.2, 47.5, 52.1, 54.9, 55.7], 'marker': 'X'},
        {'name': 'MoCoV2-supIN->SN', 'x': [20, 40, 60, 80, 100], 'y': [43.4, 49.8, 53.5, 55.1, 56.6], 'marker': 'D'},
    ]
    plot_curves(curves=data,
                xlabel='Percentage of Training Images',
                ylabel='mIoU',
                xlim=[15, 105],
                ylim=[40.0, 62.0],
                xticks=[20, 40, 60, 80, 100],
                yticks=np.arange(40, 62, 5),
                xticklabels=True,
                if_grid=True, 
                image_name='data_efficient_imagenet_resnet50.jpg')

def insseg_resnet50():
    '''
    '''
    data = [
        {'name': 'Pri3D (View)', 'x': [20, 40, 60, 80, 100], 'y': [14.5, 23.1, 28.3, 32.3, 34.3], 'marker': 'o'},
        {'name': 'Pri3D (Geo)', 'x': [20, 40, 60, 80, 100], 'y': [14.9, 22.7, 28.2, 31.8, 34.4], 'marker': '^'},
        {'name': 'Pri3D (View + Geo)', 'x': [20, 40, 60, 80, 100], 'y': [15.3, 24.1, 29.1, 32.2, 35.8], 'marker': 's'},
        {'name': 'ImageNet (IN)', 'x': [20, 40, 60, 80, 100], 'y': [14.0, 21.0, 25.8, 29.4, 32.6], 'marker': 'X'},
        {'name': 'MoCoV2-supIN->SN', 'x': [20, 40, 60, 80, 100], 'y': [15.0, 23.6, 28.2, 31.1, 33.9], 'marker': 'D'},
    ]
    plot_curves(curves=data,
                xlabel='Percentage of Training Images',
                ylabel='AP50',
                xlim=[15, 105],
                ylim=[14.0, 36.0],
                xticks=[20, 40, 60, 80, 100],
                yticks=np.arange(14, 36, 5),
                xticklabels=True,
                if_grid=True, 
                image_name='data_efficient_insseg_resnet50.jpg')

def det_resnet50():
    ''', 
    '''
    data = [
        #{'name': 'Pri3D (View)', 'x': [20, 40, 60, 80, 100], 'y': [20.4, 30.8, 37.2, 40.9, 43.7], 'marker': 'o'},
        #{'name': 'Pri3D (Geo)', 'x': [20, 40, 60, 80, 100], 'y': [21.3, 31.3, 36.6, 40.7, 44.2], 'marker': '^'},
        {'name': 'Pri3D (View + Geo)', 'x': [20, 40, 60, 80, 100], 'y': [22.0, 32.5, 38.4, 41.6, 44.5], 'marker': 's'},
        {'name': 'ImageNet (IN)', 'x': [20, 40, 60, 80, 100], 'y': [19.8, 28.0, 34.2, 38.4, 41.7], 'marker': 'X'},
        {'name': 'MoCoV2-supIN->SN', 'x': [20, 40, 60, 80, 100], 'y': [21.4, 30.7, 36.1, 39.8, 43.5], 'marker': 'D'},
    ]
    plot_curves(curves=data,
                xlabel='Percentage of Training Images',
                ylabel='AP50',
                xlim=[15, 105],
                ylim=[19, 45.0],
                xticks=[20, 40, 60, 80, 100],
                yticks=np.arange(19, 45, 5),
                xticklabels=True,
                if_grid=True, 
                image_name='data_efficient_det_resnet50_.jpg')



if __name__=='__main__':
    #scratch_resnet18()
    #scratch_resnet50()
    #imagenet_resnet18()
    #imagenet_resnet50()
    insseg_resnet50()
    det_resnet50()

