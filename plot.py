import wandb
import plotly.graph_objects as go

from evaluate import predict_convex


def plot_generator(gen_state, source, target, epoch, fig=None, row=1, col=1, plot_lst='pqt', upload=False):
  """Show the Generator plot in 2D.

  Args:
    gen_state: Lipschitz state of the generator.
    source: source dataset of shape (num_samples, num_features).
    target: source dataset of shape (num_samples, num_features).
    epoch: current epoch.
    fig: figure to update.
    row: row of the subplot.
    col: column of the subplot.
    plot_lst: list of plots to show. Can be 'p', 'q' or 't' or any combination.
    upload: whether to upload the plot to wandb.
  """
  gen = lambda inputs: predict_convex(gen_state, inputs)

  pp = source
  qq = target
  tp = gen(pp)
    
  size = 3.

  showlegend = row == col == 1
  opacity = 0.3
  data = []

  if 'q' in plot_lst:
    q_plot = go.Scatter(x=qq[:,0], y=qq[:,1], marker={'color':'green',
                                                      'size':size,
                                                      'opacity':opacity},
                        mode='markers', name='Q', showlegend=showlegend)
    data.append(q_plot)
  if 'p' in plot_lst:
      offset_x = 0.
      offset_y = 0.
      p_plot = go.Scatter(x=pp[:,0] + offset_x, y=pp[:,1] + offset_y, marker={'color':'blue',
                                                                              'size':size,
                                                                              'opacity':opacity},
                          mode='markers', name='P', showlegend=showlegend)
      data.append(p_plot)
  if 't' in plot_lst:
    tp_plot = go.Scatter(x=tp[:,0], y=tp[:,1], marker={'color':'red',
                                                      'size':size,
                                                      'opacity':opacity},
                        mode='markers', name='G#P', showlegend=showlegend)
    data.append(tp_plot)
  if fig is None:
    fig = go.Figure(data=data)
    fig.update_yaxes(
      scaleratio = 1,
      title=dict(text='y')
    )
    fig.update_xaxes(
      scaleanchor = "y",
      title=dict(text='x'),
      scaleratio = 1,
    )
    fig.update_layout(font=dict(size=16),
                      legend= {'itemsizing': 'constant'},
                      autosize=True, width=500, height=500)
    fig.write_image("figures/G#P.png")
    if upload:
      wandb.log({"G#P": wandb.Plotly(fig)})
