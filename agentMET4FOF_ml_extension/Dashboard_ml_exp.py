"""
Provides functions to obtain data from experiments folder (default is MLEXP)
To be rendered on the tab of the dashboard for visualizing and comparing ML experiments
"""

import dash_html_components as html
import dash_core_components as dcc
import dash_table
from dash.exceptions import PreventUpdate
from agentMET4FOF.agentMET4FOF.dashboard import LayoutHelper
import pandas as pd

from analyse_ml_results import load_ml_exp_details, load_pd_full, groupby_results
from dash.dependencies import Input, Output, State
import numpy as np
from math import log10, floor
from agentMET4FOF.agentMET4FOF.dashboard.Dashboard_layout_base import Dashboard_Layout_Base

column_rename_map = {"data": "Datastream Agents", "model": "Model Agents",
                     "random_state": "Random Seed",
                     "perf_name": "Performance", "perf_score": "Score",
                     "run_name": "Experiment Name",
                     "date": "Date",
                     "data_params": "Data Parameters",
                     "model_params": "Model Parameters"
                     }


def rename_column(df_columns):
    renamed_df_columns = [column_rename_map[column] for column in df_columns]
    return renamed_df_columns


class Dashboard_ML_Experiment(Dashboard_Layout_Base):


    def set_layout_name(self,id="ml-exp", title="ML Experiment"):
        self.id = id
        self.title=title

    def get_experiments_list(self):
        try:
            experiment_list = load_ml_exp_details()
            experiment_list['date'] = pd.DatetimeIndex(experiment_list['date']).strftime("%d-%m-%Y, %H:%M:%S")

        except Exception as e:
            experiment_list = {}
        return experiment_list

    def get_layout(self):
        #update with the latest experiments list
        experiments_df=self.get_experiments_list()

        #body
        return html.Div(className="row",children=[
            #main panel
            html.Div(className="col s8", children=[
                    html.Div(className="card", children=[
                       html.Div(className="card-content", children=[
                               html.Span(className="card-title", children=["Results"]),
                                           html.Div(className="row", children = [
                                                        html.Div(className="col", children=[
                                                            LayoutHelper.html_button(icon="select_all",text="Select All", id="select-all-button")
                                                        ]),
                                                        html.Div(className="col", children=[
                                                            LayoutHelper.html_button(icon="tab_unselected",text="Unselect All", id="unselect-all-button")
                                                        ]),
                                                    ])
                        ]),
                       html.Div(className="card-action", id="chains-div", children=[
                        LayoutHelper.create_params_table(table_name="chains-table",
                                                         data={},
                                                            editable=True,
                                                            filter_action="native",
                                                            sort_action="native",
                                                            sort_mode="multi",
                                                            row_selectable="multi",
                                                            selected_rows=[],
                                                         ),
                        ])

                    ]),
                    html.Div(className="card", id="compare-graph-div", children=[]),
                    html.Div(id="placeholder-select",style={"opacity":0}),
            ]),

            #side panel
            html.Div(className="col s4", children=[
                html.Div(className="card blue lighten-4", children= [
                    html.Div(className="card-content", children=[

                        html.Span(className="card-title",style={'margin-top': '20px'}, children=[
                             "ML Experiments"
                        ]),
                        LayoutHelper.create_params_table(table_name="experiment-table",
                                                         data=experiments_df,
                                                            editable=True,
                                                            filter_action="native",
                                                            sort_action="native",
                                                            sort_mode="multi",
                                                            row_selectable="multi",
                                                            selected_rows=[],
                                                         rename_map=column_rename_map
                                                         ),
                        html.H6([""],id="experiment-placeholder-selected-rows")
                    ])
                ]),

                html.Div(className="card blue lighten-4", children= [
                    html.Div(className="card-content", children=[

                        html.Span(className="card-title",style={'margin-top': '20px'}, children=[
                             "Data and Model Parameters"
                        ]),
                        html.Div(id="pipeline-div",children=
                        LayoutHelper.create_params_table(table_name="pipeline-table",
                                                        data={},
                                                        # editable=True,
                                                        filter_action="native",
                                                        sort_action="native",
                                                        sort_mode="multi",
                                                        # row_selectable="multi",
                                                        # selected_rows=[],
                                                         )
                        )
                    ])
                ]),


            ]),
        ])

    def prepare_callbacks(self,app):
        app.ml_experiments = []
        app.aggregated_chain_results ={}

        @app.callback(
            [Output('chains-table', "selected_rows"),],
            [Input('select-all-button', 'n_clicks_timestamp'),
             Input('unselect-all-button', 'n_clicks_timestamp')],
            [State('chains-table', "derived_virtual_data"),
             ]
        )
        def unselect_all(select_timestamp,unselect_timestamp, rows):
            if select_timestamp is None and unselect_timestamp is None:
                return [[]]
            res = "SELECT"
            if unselect_timestamp is None:
                res = "SELECT"
            elif select_timestamp is None:
                res = "UNSELECT"
            elif select_timestamp > unselect_timestamp:
                res = "SELECT"
            else:
                res = "UNSELECT"

            if res == "SELECT":
                return [[i for i in range(len(rows))]]
            elif res == "UNSELECT":
                return [[]]

        @app.callback(
            [
             Output('pipeline-div', "children"),
             Output('chains-div', "children"),
             ],
            [Input('experiment-table', "derived_virtual_data"),
             Input('experiment-table', "derived_virtual_selected_rows")])
        def update_experiment_table(rows, derived_virtual_selected_rows):
            if derived_virtual_selected_rows is None:
                derived_virtual_selected_rows = []
                raise PreventUpdate

            ml_experiments = []
            pipeline_details = []
            chain_results =[]
            if len(derived_virtual_selected_rows) > 0:
                aggregated_chain_results = load_pd_full(run_names=[rows[i]["run_name"] for i in derived_virtual_selected_rows],
                                             dates=[rows[i]["date"] for i in derived_virtual_selected_rows])


                # create pipeline table which mashes (data-dataparams-trainsize, model-modelparams)
                pipeline_table = aggregated_chain_results.copy()
                #
                model_uniques = (pipeline_table["model"].unique())
                data_uniques = (pipeline_table["data"].unique())
                pipeline_table["model_sign"] = 0
                pipeline_table["data_sign"] = 0

                for model in model_uniques:
                    subset = pipeline_table[pipeline_table["model"] == model]
                    model_param_uniques = (subset["model_params"].unique())
                    for id_, model_param in enumerate(model_param_uniques):
                        pipeline_table.loc[(pipeline_table["model"] == model) & (pipeline_table["model_params"] == model_param), "model_sign"] = model+("("+str(id_)+")" if len(model_param_uniques) > 1 else "")

                for data in data_uniques:
                    subset = pipeline_table[pipeline_table["data"] == data]
                    data_param_uniques = (subset["data_params"].unique())
                    for id_, data_param in enumerate(data_param_uniques):
                        pipeline_table.loc[(pipeline_table["data"] == data) & (pipeline_table["data_params"] == data_param), "data_sign"] = data+("("+str(id_)+")" if len(data_param_uniques) > 1 else "")

                pipeline_table["data"] = pipeline_table["data_sign"]
                pipeline_table["model"] = pipeline_table["model_sign"]

                pipeline_table = pipeline_table.drop(["perf_name","perf_score","random_state"], axis="columns").astype(str)

                data_unique_table = pipeline_table[["data","data_params"]].copy().drop_duplicates()
                model_unique_table = pipeline_table[["model", "model_params"]].copy().drop_duplicates()
                model_unique_table = model_unique_table.sort_values(by=["model"],axis=0)
                data_unique_table = data_unique_table.sort_values(by=["data"], axis=0)

                # add unique signature to the main results
                aggregated_chain_results["model"] = pipeline_table["model_sign"]
                aggregated_chain_results["data"] = pipeline_table["data_sign"]
                aggregated_chain_results = aggregated_chain_results.drop(["data_params","model_params"], axis="columns")

                app.aggregated_chain_results = aggregated_chain_results
                selected_rows_chains = np.arange(0, aggregated_chain_results.shape[0])
                aggregated_chain_results = aggregated_chain_results.applymap(round_sig)

                # create main results table
                chains_table = dash_table.DataTable(data=aggregated_chain_results.to_dict('records'),
                                                         columns=[{'id': c, 'name': column_rename_map[c]} for c in aggregated_chain_results.columns],
                                                         style_data={
                                                                'whiteSpace': 'normal',
                                                                'height': 'auto',
                                                                'text-align':'left'
                                                            },
                                                         row_selectable="multi",
                                                         selected_rows=selected_rows_chains,
                                                         filter_action="native",
                                                         sort_action="native",
                                                         sort_mode="multi",
                                                         id="chains-table",
                                                        style_header={'backgroundColor': 'rgb(66, 135, 245)', 'color': 'white'},
                                                         )
                # create pipeline table
                data_pipeline_table = dash_table.DataTable(data=data_unique_table.to_dict('records'),
                                                         columns=[{'id': c, 'name': column_rename_map[c]} for c in data_unique_table.columns],
                                                         style_data={
                                                                'whiteSpace': 'normal',
                                                                'height': 'auto',
                                                                'text-align':'left'
                                                            },
                                                         id="pipeline-table",
                                                           style_header={'backgroundColor': 'rgb(66, 135, 245)', 'color': 'white'},
                                                         )
                model_pipeline_table = dash_table.DataTable(data=model_unique_table.to_dict('records'),
                                                         columns=[{'id': c, 'name': column_rename_map[c]} for c in model_unique_table.columns],
                                                         style_data={
                                                                'whiteSpace': 'normal',
                                                                'height': 'auto',
                                                                'text-align':'left'
                                                            },
                                                            style_header={'backgroundColor': 'rgb(66, 135, 245)',
                                                                          'color': 'white'},
                                                         )
            else:
                chains_table = []
                data_pipeline_table = []
                model_pipeline_table = []



            return [[data_pipeline_table,model_pipeline_table],chains_table]

        def round_sig(x, sig=2):
            try:
                if isinstance(x, int):
                    return x
                else:
                    return round(x, sig-int(floor(log10(abs(x))))-1)
            except:
                return x

        @app.callback(
            [Output('compare-graph-div', "children")],
            [Input('chains-table', "derived_virtual_data"),
             Input('chains-table', "derived_virtual_selected_rows")]
            )
        def update_experiment_table2(derived_virtual_data, derived_virtual_selected_rows):
            if derived_virtual_selected_rows is None or len(derived_virtual_selected_rows) == 0:
                derived_virtual_selected_rows = []
                return [""]

            aggregated_chain_results = app.aggregated_chain_results
            if len(aggregated_chain_results) == 0:
                return [""]

            select_aggregated_chain_results = pd.DataFrame([derived_virtual_data[i] for i in derived_virtual_selected_rows])

            pd_mean, pd_sem = groupby_results(select_aggregated_chain_results,groupby_cols=["data","model"], perf_columns=["perf_score"])
            pd_mean = pd_mean.applymap(round_sig)
            pd_sem = pd_sem.applymap(round_sig)
            pd_mean["group_graphs"] = pd_mean["data"]+" ("+pd_mean["perf_name"]+")"
            final_graphs = []
            for column, group_graph in enumerate(pd_mean["group_graphs"].unique()):

                subset_pd = pd_mean[pd_mean["group_graphs"]==group_graph]

                final_graph = dcc.Graph(
                        id=str(column) + '--row-ids',
                        figure={
                            'data': [
                                {
                                    'x': subset_pd['model'],
                                    'y': subset_pd['perf_score'],
                                    'type': 'bar',
                                    # 'marker': {'color': colors},
                                }
                            ],
                            'layout': {
                                'xaxis': {'automargin': True},
                                'yaxis': {
                                    'automargin': True,
                                    'title': {'text': subset_pd["perf_name"]}
                                },
                                'height': 350,
                                'margin': {'t': 45, 'l': 10, 'r': 10},
                                'title': group_graph
                            },
                        },
                    )
                final_graphs.append(final_graph)
            # print(final_graphs)
            return [final_graphs]
            # return final_graphs
        return app

