import plotly.graph_objects as go
import os


def plot_causal_effect(combined_scales_df,
                       scales, 
                       image_name, 
                       out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig = go.Figure()
    scale_columns = [f"scale {scale}" for scale in scales]
    for i, row in combined_scales_df.iterrows():
        y = [row[col] for col in scale_columns]
        x = scales
        # plot lines for each node
        fig.add_trace(go.Line(x=x, y=y, mode='lines+markers', 
                            # set color based on status
                            line=dict(color="green" if row["status"] == "in_circuit" else "orange"),
                            hovertext=f"Node: {row['node']}, Status: {row['status']}",
                            # define legend only for color, not for line
                            showlegend=False,
                            ),
                    )
    fig.update_layout(xaxis_title="Scale", yaxis_title="Causal Effect")
    # make legend for color
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color="green"), name="in_circuit"))
    fig.add_trace(go.Scatter(x=[None], y=[None], mode='markers', marker=dict(color="orange"), name="not_in_circuit"))
    # make background transparent and remove grid
    fig.update_layout(template="plotly_white")
    # remove grid
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    # decrease margin
    fig.update_layout(margin=dict(l=70, r=70, t=70, b=70))
    # increase font size
    fig.update_layout(font=dict(size=16))
    # add title
    fig.update_layout(title=image_name)
    fig.show()
    # save to file as pdf with same width and height
    fig.write_image(f"{out_dir}/{image_name}.png")