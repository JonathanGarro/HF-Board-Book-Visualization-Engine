import pandas as pd
import plotly.graph_objects as go
import os
import textwrap

# grant id filter settings
ENABLE_GRANT_ID_FILTER = True  # set to False to disable filtering

# list of grant ids to exclude from all visualizations
EXCLUDED_GRANT_IDS = [
    '2025-06314-PRO',
    '2025-06318-PRO'
]

# visualization settings
GENERATE_TREEMAP = True
GENERATE_SUNBURST = True
GENERATE_SUNBURST_THREE_LEVEL = True
GENERATE_SUNBURST_FOUR_LEVEL = True
GENERATE_CSV = True
GENERATE_TREEMAP_TOP_GRANTEES = True

# threshold setting for organization filtering (minimum amount in dollars)
MIN_ORG_AMOUNT = 1000000  # $1M
GENERATE_2M_VERSIONS = True

# program colors
program_colors = {
    'EDUCATION': '#1A254E',
    'ENVIRONMENT': '#778218',
    'GENDER EQUITY & GOVERNANCE': '#E89829',
    'PERFORMING ARTS': '#4A0F3E',
    'U.S. DEMOCRACY': '#3E006C',
    'PHILANTHROPY': '#214240',
    'CULTURE, RACE, & EQUITY': '#C15811',
    'ECONOMY AND SOCIETY INITIATIVE': '#184319',
    'SPECIAL PROJECTS': '#184319',
    'SBAC': '#414B3F',
    'CYBER': '#414B3F'
}

# create outputs directory if it doesn't exist
os.makedirs('outputs', exist_ok=True)


# helper functions
def wrap_text(text, width=20):
    """wrap text to specified width, replacing spaces with <br> for html line breaks"""
    if pd.isna(text) or not isinstance(text, str):
        return text
    wrapped = textwrap.fill(text, width=width)
    return wrapped.replace('\n', '<br>')


def lighten_color(hex_color, factor=0.4):
    """lighten a hex color by mixing it with white"""
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    return f'#{r:02x}{g:02x}{b:02x}'


def calculate_angles_and_filter(df_grouped, min_amount=1000000):
    """
    filter organizations below minimum amount threshold ($1M by default)
    and group the rest into "other". GUARANTEES every strategy has at least one entry.
    """
    result_rows = []

    for (program, strategy) in df_grouped[['Program', 'Strategy']].drop_duplicates().values:
        strategy_data = df_grouped[(df_grouped['Program'] == program) &
                                   (df_grouped['Strategy'] == strategy)].copy()

        kept_orgs = strategy_data[strategy_data['Amount'] >= min_amount].copy()
        filtered_out = strategy_data[strategy_data['Amount'] < min_amount].copy()

        if len(kept_orgs) > 0:
            result_rows.append(kept_orgs[['Program', 'Strategy', 'Organization Name', 'Amount']])

        if len(filtered_out) > 0:
            other_amount = filtered_out['Amount'].sum()
            other_count = len(filtered_out)
            other_row = pd.DataFrame({
                'Program': [program],
                'Strategy': [strategy],
                'Organization Name': [f'Other ({other_count})'],
                'Amount': [other_amount]
            })
            result_rows.append(other_row)
        elif len(kept_orgs) == 0:
            print(f"WARNING: Strategy {program}/{strategy} has no organizations!")
            dummy_row = pd.DataFrame({
                'Program': [program],
                'Strategy': [strategy],
                'Organization Name': ['No organizations'],
                'Amount': [0]
            })
            result_rows.append(dummy_row)

    if not result_rows:
        return pd.DataFrame()

    result_df = pd.concat(result_rows, ignore_index=True)
    print(
        f"Filtering complete: {len(df_grouped)} orgs -> {len(result_df)} entries ({len(result_df[result_df['Organization Name'].str.startswith('Other')])} Other groups)")
    return result_df


def calculate_angles_and_filter_four_level(df_grouped, min_amount=1000000):
    """
    filter organizations below minimum amount threshold ($1M by default)
    and group the rest into "other". GUARANTEES every substrategy has at least one entry.
    """
    result_rows = []

    for (program, strategy, substrategy) in df_grouped[
        ['Program', 'Strategy', 'Substrategy']].drop_duplicates().values:
        substrategy_data = df_grouped[(df_grouped['Program'] == program) &
                                      (df_grouped['Strategy'] == strategy) &
                                      (df_grouped['Substrategy'] == substrategy)].copy()

        kept_orgs = substrategy_data[substrategy_data['Amount'] >= min_amount].copy()
        filtered_out = substrategy_data[substrategy_data['Amount'] < min_amount].copy()

        if len(kept_orgs) > 0:
            result_rows.append(kept_orgs[['Program', 'Strategy', 'Substrategy', 'Organization Name', 'Amount']])

        if len(filtered_out) > 0:
            other_amount = filtered_out['Amount'].sum()
            other_count = len(filtered_out)
            other_row = pd.DataFrame({
                'Program': [program],
                'Strategy': [strategy],
                'Substrategy': [substrategy],
                'Organization Name': [f'Other ({other_count})'],
                'Amount': [other_amount]
            })
            result_rows.append(other_row)
        elif len(kept_orgs) == 0:
            print(f"WARNING: Substrategy {program}/{strategy}/{substrategy} has no organizations!")

    return pd.concat(result_rows, ignore_index=True) if result_rows else pd.DataFrame()


# data loading
df = pd.read_csv('00OUf000008PyafMAC_mapped.csv')

# apply grant id filter if enabled
if ENABLE_GRANT_ID_FILTER and len(EXCLUDED_GRANT_IDS) > 0:
    initial_count = len(df)
    grant_id_column = 'Request: Reference Number'

    if grant_id_column in df.columns:
        # show which grants are being excluded
        excluded_rows = df[df[grant_id_column].isin(EXCLUDED_GRANT_IDS)]
        if len(excluded_rows) > 0:
            print(f"\nexcluding {len(excluded_rows)} grants:")
            for idx, row in excluded_rows.iterrows():
                print(
                    f"  - {row[grant_id_column]}: {row.get('Organization: Organization Name', 'N/A')} (${row.get('Amount', 0):,.0f})")

        # apply filter
        df = df[~df[grant_id_column].isin(EXCLUDED_GRANT_IDS)]

        filtered_count = initial_count - len(df)
        print(f"\ngrant id filter applied: excluded {filtered_count} grants")
        print(f"remaining grants: {len(df)}")
    else:
        print(f"WARNING: could not find column '{grant_id_column}'. available columns: {list(df.columns)}")
        print("grant id filter not applied")
else:
    print("grant id filter disabled")

# filter out rows with missing program or amount, but keep rows with missing strategy
df_clean = df[['Program', 'Strategy', 'Amount']].copy()
df_clean = df_clean[df_clean['Program'].notna() & df_clean['Amount'].notna()]

# replace null/blank strategies with "unspecified"
df_clean['Strategy'] = df_clean['Strategy'].fillna('unspecified')

# group by program and strategy, summing amounts
df_grouped = df_clean.groupby(['Program', 'Strategy'], as_index=False)['Amount'].sum()

# calculate total amount for percentage calculation
total_amount = df_grouped['Amount'].sum()

# treemap visualization
if GENERATE_TREEMAP:
    print("generating treemap...")

    labels = []
    parents = []
    values = []
    colors = []
    custom_text = []

    for program in df_grouped['Program'].unique():
        labels.append(program)
        parents.append('')
        values.append(0)
        colors.append(program_colors.get(program, '#CCCCCC'))
        custom_text.append(program)

    for idx, row in df_grouped.iterrows():
        program = row['Program']
        strategy = row['Strategy']
        amount = row['Amount']

        unique_label = f"{program}_{strategy}_{idx}"
        labels.append(unique_label)
        parents.append(program)
        values.append(amount)
        colors.append(lighten_color(program_colors.get(program, '#CCCCCC')))

        if amount / total_amount < 0.01 or strategy == 'unspecified':
            custom_text.append('')
        else:
            custom_text.append(wrap_text(strategy))

    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        text=custom_text,
        textinfo='text',
        marker=dict(
            colors=colors,
            line=dict(width=2, color='white')
        ),
        textposition='middle center',
        textfont=dict(size=12),
        hovertemplate='<b>%{text}</b><br>Amount: %{value:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title='Grants by Program and Strategy',
        width=1600,
        height=1000,
        font_size=11
    )

    output_path = 'outputs/treemap_program_strategy.png'
    fig.write_image(output_path)
    print(f"treemap saved to {output_path}")

    output_path_html = 'outputs/treemap_program_strategy.html'
    fig.write_html(output_path_html)
    print(f"interactive treemap saved to {output_path_html}")

# sunburst visualization
if GENERATE_SUNBURST:
    print("generating sunburst...")

    ids = []
    labels = []
    parents = []
    values = []
    colors = []

    for program in df_grouped['Program'].unique():
        ids.append(program)
        labels.append(program)
        parents.append('')
        values.append(0)
        colors.append(program_colors.get(program, '#CCCCCC'))

    for idx, row in df_grouped.iterrows():
        program = row['Program']
        strategy = row['Strategy']
        amount = row['Amount']

        strategy_id = f"{program}_{strategy}_{idx}"
        ids.append(strategy_id)
        labels.append('' if strategy.lower() == 'unspecified' or amount / total_amount < 0.01 else strategy)
        parents.append(program)
        values.append(amount)
        colors.append(lighten_color(program_colors.get(program, '#CCCCCC')))

    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(colors=colors, line=dict(width=2, color='white')),
        textinfo='label+percent entry',
        insidetextorientation='radial',
        textfont=dict(size=12),
        hovertemplate='<b>%{label}</b><br>Amount: %{value:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title='Grants by Program and Strategy',
        width=1200,
        height=1200,
        font_size=12
    )

    output_path = 'outputs/sunburst_program_strategy.png'
    fig.write_image(output_path)
    print(f"sunburst chart saved to {output_path}")

    output_path_svg = 'outputs/sunburst_program_strategy.svg'
    fig.write_image(output_path_svg)
    print(f"sunburst chart svg saved to {output_path_svg}")

    output_path_html = 'outputs/sunburst_program_strategy.html'
    fig.write_html(output_path_html)
    print(f"interactive sunburst chart saved to {output_path_html}")

# complete three-level sunburst section with amount filtering
if GENERATE_SUNBURST_THREE_LEVEL:
    print("generating three-level sunburst (program > strategy > top orgs with amount filtering)...")

    df_three_level = df[['Program', 'Strategy', 'Organization: Organization Name', 'Amount']].copy()
    df_three_level = df_three_level.rename(columns={'Organization: Organization Name': 'Organization Name'})
    df_three_level['Strategy'] = df_three_level['Strategy'].fillna('unspecified')
    df_three_level = df_three_level[df_three_level['Program'].notna() &
                                    df_three_level['Organization Name'].notna() &
                                    df_three_level['Amount'].notna()]

    df_three_grouped = df_three_level.groupby(['Program', 'Strategy', 'Organization Name'],
                                              as_index=False)['Amount'].sum()

    print(f"filtering organizations with amounts less than ${MIN_ORG_AMOUNT:,.0f}...")
    df_three_final = calculate_angles_and_filter(df_three_grouped, min_amount=MIN_ORG_AMOUNT)

    df_three_final['Is_Other'] = df_three_final['Organization Name'].str.startswith('Other').astype(int)
    df_three_final = df_three_final.sort_values(['Program', 'Strategy', 'Is_Other', 'Amount'],
                                                ascending=[True, True, True, False])
    df_three_final = df_three_final.drop('Is_Other', axis=1)

    ids, labels, parents, values, colors = [], [], [], [], []

    for program in df_three_final['Program'].unique():
        ids.append(program)
        labels.append(program)
        parents.append('')
        values.append(0)
        colors.append(program_colors.get(program, '#CCCCCC'))

    for (program, strategy) in df_three_final[['Program', 'Strategy']].drop_duplicates().values:
        strategy_id = f"{program}_{strategy}"
        ids.append(strategy_id)
        labels.append('' if strategy.lower() == 'unspecified' else strategy)
        parents.append(program)
        values.append(0)
        colors.append(lighten_color(program_colors.get(program, '#CCCCCC'), factor=0.3))

    for idx, row in df_three_final.iterrows():
        program, strategy, org, amount = row['Program'], row['Strategy'], row['Organization Name'], row['Amount']
        org_id = f"{program}_{strategy}_{org}_{idx}"
        strategy_id = f"{program}_{strategy}"
        ids.append(org_id)
        labels.append('' if org.startswith('Other') else wrap_text(org, 15))
        parents.append(strategy_id)
        values.append(amount)
        colors.append(
            '#CCCCCC' if org.startswith('Other') else lighten_color(program_colors.get(program, '#CCCCCC'), factor=0.6))

    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        textinfo='label+percent entry',
        marker=dict(colors=colors, line=dict(width=2, color='white')),
        insidetextorientation='radial',
        hovertemplate='<b>%{label}</b><br>Amount: %{value:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title=f'Grants by Program, Strategy, and Organizations (min ${MIN_ORG_AMOUNT / 1000000:.0f}M per org)',
        width=1400,
        height=1400,
        font_size=11
    )

    output_path = 'outputs/sunburst_three_level.png'
    fig.write_image(output_path)
    print(f"three-level sunburst chart saved to {output_path}")

    output_path_html = 'outputs/sunburst_three_level.html'
    fig.write_html(output_path_html)
    print(f"interactive three-level sunburst chart saved to {output_path_html}")

    original_count = len(df_three_grouped)
    filtered_count = len(df_three_final[~df_three_final['Organization Name'].str.startswith('Other')])
    print(f"organizations: {original_count} original, {filtered_count} after filtering")

    # create directory for individual program sunbursts
    program_dir = 'outputs/sunburst_three_level_programs'
    os.makedirs(program_dir, exist_ok=True)
    print(f"\ngenerating zoomed sunbursts for each program with $1M threshold...")

    AMOUNT_THRESHOLD = 1000000  # $1 million

    # generate a zoomed sunburst for each program
    for program in df_three_grouped['Program'].unique():
        program_data = df_three_grouped[df_three_grouped['Program'] == program].copy()

        # apply the $1m threshold filtering for each strategy
        filtered_program_data = []
        strategies_sorted = sorted(program_data['Strategy'].unique())

        for strategy in strategies_sorted:
            strategy_data = program_data[program_data['Strategy'] == strategy]

            large_orgs = strategy_data[strategy_data['Amount'] >= AMOUNT_THRESHOLD].copy()
            small_orgs = strategy_data[strategy_data['Amount'] < AMOUNT_THRESHOLD]

            large_orgs = large_orgs.sort_values('Amount', ascending=False)

            if len(large_orgs) > 0:
                large_orgs['_sort_order'] = range(len(large_orgs))
                filtered_program_data.append(large_orgs)

            if len(small_orgs) > 0:
                other_amount = small_orgs['Amount'].sum()
                other_count = len(small_orgs)
                other_row = pd.DataFrame({
                    'Program': [program],
                    'Strategy': [strategy],
                    'Organization Name': [f'Other ({other_count})'],
                    'Amount': [other_amount],
                    '_sort_order': [999999]  # ensure "other" comes last
                })
                filtered_program_data.append(other_row)

        if len(filtered_program_data) > 0:
            program_data_filtered = pd.concat(filtered_program_data, ignore_index=True)
        else:
            continue

        program_data_filtered = program_data_filtered.sort_values(
            ['Strategy', '_sort_order'],
            ascending=[True, True]
        )

        program_data_filtered = program_data_filtered.drop('_sort_order', axis=1)

        # create lists for this program's sunburst (without program level)
        prog_ids = []
        prog_labels = []
        prog_parents = []
        prog_values = []
        prog_colors = []

        # add a hidden root node with empty label
        prog_ids.append('root')
        prog_labels.append('')
        prog_parents.append('')
        prog_values.append(0)
        prog_colors.append('#FFFFFF')  # white color for hidden root

        # level 1: strategies
        for strategy in program_data_filtered['Strategy'].unique():
            strategy_id = f"{program}_{strategy}"
            prog_ids.append(strategy_id)
            prog_labels.append('' if strategy.lower() == 'unspecified' else strategy)
            prog_parents.append('root')
            prog_values.append(0)
            prog_colors.append(lighten_color(program_colors.get(program, '#CCCCCC'), factor=0.3))

        # level 2: organizations (order preserved by the dataframe)
        for idx, row in program_data_filtered.iterrows():
            strategy = row['Strategy']
            org = row['Organization Name']
            amount = row['Amount']

            org_id = f"{program}_{strategy}_{org}_{idx}"
            strategy_id = f"{program}_{strategy}"

            prog_ids.append(org_id)
            prog_labels.append(org)
            prog_parents.append(strategy_id)
            prog_values.append(amount)

            if org.startswith('Other'):
                prog_colors.append('#CCCCCC')
            else:
                prog_colors.append(lighten_color(program_colors.get(program, '#CCCCCC'), factor=0.6))

        # IMPORTANT FIX: keep Plotly from re-sorting by value so "Other" stays last
        fig_program = go.Figure(go.Sunburst(
            ids=prog_ids,
            labels=prog_labels,
            parents=prog_parents,
            values=prog_values,
            sort=False,  # <—— preserve the order we've provided
            marker=dict(
                colors=prog_colors,
                line=dict(width=2, color='white')
            ),
            insidetextorientation='radial',
            hovertemplate='<b>%{label}</b><br>Amount: $%{value:,.0f}<extra></extra>',
            textfont=dict(size=11)
        ))

        fig_program.update_layout(
            title=f'{program} - Strategies and Organizations ($1M+ threshold)',
            width=1400,
            height=1400,
            font_size=11
        )

        program_filename = program.replace(' ', '_').replace('&', 'and').replace(':', '').lower()

        output_path_png = f'{program_dir}/{program_filename}_zoomed_1m.png'
        fig_program.write_image(output_path_png)

        output_path_svg = f'{program_dir}/{program_filename}_zoomed_1m.svg'
        fig_program.write_image(output_path_svg)

        output_path_html = f'{program_dir}/{program_filename}_zoomed_1m.html'
        fig_program.write_html(output_path_html)

        original_org_count = len(program_data[~program_data['Organization Name'].str.startswith('Other')])
        shown_orgs = len(program_data_filtered[~program_data_filtered['Organization Name'].str.startswith('Other')])
        other_orgs = original_org_count - shown_orgs
        print(f"saved zoomed sunburst for {program}: {shown_orgs} orgs >= $1M shown, {other_orgs} grouped as 'Other'")

    # generate 2m threshold versions if enabled
    if GENERATE_2M_VERSIONS:
        # create subdirectory for 2m versions
        program_2m_dir = 'outputs/sunburst_three_level_programs/2M_versions'
        os.makedirs(program_2m_dir, exist_ok=True)
        print(f"\ngenerating $2M threshold versions for crowded programs...")

        TWO_MILLION_THRESHOLD = 2000000  # $2 million
        ONE_MILLION_THRESHOLD = 1000000  # $1 million

        # generate a zoomed sunburst for each program with 3-tier bucketing
        for program in df_three_grouped['Program'].unique():
            program_data = df_three_grouped[df_three_grouped['Program'] == program].copy()

            # apply the 3-tier threshold filtering for each strategy
            filtered_program_data = []
            strategies_sorted = sorted(program_data['Strategy'].unique())

            for strategy in strategies_sorted:
                strategy_data = program_data[program_data['Strategy'] == strategy]

                # three buckets: >= $2m, $1m-$2m, < $1m
                large_orgs = strategy_data[strategy_data['Amount'] >= TWO_MILLION_THRESHOLD].copy()
                medium_orgs = strategy_data[(strategy_data['Amount'] >= ONE_MILLION_THRESHOLD) &
                                            (strategy_data['Amount'] < TWO_MILLION_THRESHOLD)]
                small_orgs = strategy_data[strategy_data['Amount'] < ONE_MILLION_THRESHOLD]

                # sort large orgs by amount descending
                large_orgs = large_orgs.sort_values('Amount', ascending=False)

                # add large orgs (>= $2m) individually
                if len(large_orgs) > 0:
                    large_orgs['_sort_order'] = range(len(large_orgs))
                    filtered_program_data.append(large_orgs)

                # group medium orgs ($1m-$2m) into one segment
                if len(medium_orgs) > 0:
                    medium_amount = medium_orgs['Amount'].sum()
                    medium_count = len(medium_orgs)
                    medium_row = pd.DataFrame({
                        'Program': [program],
                        'Strategy': [strategy],
                        'Organization Name': [f'$1M-$2M ({medium_count})'],
                        'Amount': [medium_amount],
                        '_sort_order': [999998]  # ensure this comes before "other"
                    })
                    filtered_program_data.append(medium_row)

                # group small orgs (< $1m) into "other"
                if len(small_orgs) > 0:
                    other_amount = small_orgs['Amount'].sum()
                    other_count = len(small_orgs)
                    other_row = pd.DataFrame({
                        'Program': [program],
                        'Strategy': [strategy],
                        'Organization Name': [f'Other ({other_count})'],
                        'Amount': [other_amount],
                        '_sort_order': [999999]  # ensure "other" comes last
                    })
                    filtered_program_data.append(other_row)

            if len(filtered_program_data) > 0:
                program_data_filtered = pd.concat(filtered_program_data, ignore_index=True)
            else:
                continue

            # sort by strategy and sort order
            program_data_filtered = program_data_filtered.sort_values(
                ['Strategy', '_sort_order'],
                ascending=[True, True]
            )

            program_data_filtered = program_data_filtered.drop('_sort_order', axis=1)

            # create lists for this program's sunburst (without program level)
            prog_ids = []
            prog_labels = []
            prog_parents = []
            prog_values = []
            prog_colors = []

            # add a hidden root node with empty label
            prog_ids.append('root')
            prog_labels.append('')
            prog_parents.append('')
            prog_values.append(0)
            prog_colors.append('#FFFFFF')  # white color for hidden root

            # level 1: strategies
            for strategy in program_data_filtered['Strategy'].unique():
                strategy_id = f"{program}_{strategy}"
                prog_ids.append(strategy_id)
                prog_labels.append('' if strategy.lower() == 'unspecified' else strategy)
                prog_parents.append('root')
                prog_values.append(0)
                prog_colors.append(lighten_color(program_colors.get(program, '#CCCCCC'), factor=0.3))

            # level 2: organizations (order preserved by the dataframe)
            for idx, row in program_data_filtered.iterrows():
                strategy = row['Strategy']
                org = row['Organization Name']
                amount = row['Amount']

                org_id = f"{program}_{strategy}_{org}_{idx}"
                strategy_id = f"{program}_{strategy}"

                prog_ids.append(org_id)
                prog_labels.append(org)
                prog_parents.append(strategy_id)
                prog_values.append(amount)

                # color coding: gray for grouped segments, lighter color for individual orgs
                if org.startswith('Other') or org.startswith('$1M-$2M'):
                    prog_colors.append('#CCCCCC')
                else:
                    prog_colors.append(lighten_color(program_colors.get(program, '#CCCCCC'), factor=0.6))

            # create the figure with sort=false to preserve our custom order
            fig_program = go.Figure(go.Sunburst(
                ids=prog_ids,
                labels=prog_labels,
                parents=prog_parents,
                values=prog_values,
                sort=False,  # preserve the order we've provided
                marker=dict(
                    colors=prog_colors,
                    line=dict(width=2, color='white')
                ),
                insidetextorientation='radial',
                hovertemplate='<b>%{label}</b><br>Amount: $%{value:,.0f}<extra></extra>',
                textfont=dict(size=11)
            ))

            fig_program.update_layout(
                title=f'{program} - Strategies and Organizations ($2M+ individual, $1M-$2M grouped)',
                width=1400,
                height=1400,
                font_size=11
            )

            # save the files
            program_filename = program.replace(' ', '_').replace('&', 'and').replace(':', '').lower()

            output_path_png = f'{program_2m_dir}/{program_filename}_2m_threshold.png'
            fig_program.write_image(output_path_png)

            output_path_svg = f'{program_2m_dir}/{program_filename}_2m_threshold.svg'
            fig_program.write_image(output_path_svg)

            output_path_html = f'{program_2m_dir}/{program_filename}_2m_threshold.html'
            fig_program.write_html(output_path_html)

            # calculate and print statistics
            original_org_count = len(program_data[~program_data['Organization Name'].str.startswith('Other')])
            shown_orgs = len(program_data_filtered[
                                 (~program_data_filtered['Organization Name'].str.startswith('Other')) &
                                 (~program_data_filtered['Organization Name'].str.startswith('$1M-$2M'))
                                 ])
            medium_orgs_count = len(program_data[
                                        (program_data['Amount'] >= ONE_MILLION_THRESHOLD) &
                                        (program_data['Amount'] < TWO_MILLION_THRESHOLD)
                                        ])
            other_orgs = original_org_count - shown_orgs - medium_orgs_count

            print(f"saved $2m version for {program}: {shown_orgs} orgs >= $2M shown individually, "
                  f"{medium_orgs_count} in $1M-$2M group, {other_orgs} in 'Other'")

# four-level sunburst visualization (program > strategy > substrategy > top orgs with amount filtering)
if GENERATE_SUNBURST_FOUR_LEVEL:
    print("generating four-level sunburst (program > strategy > substrategy > top orgs with amount filtering)...")

    df_four_level = df[['Program', 'Strategy', 'Substrategy', 'Organization: Organization Name', 'Amount']].copy()
    df_four_level = df_four_level.rename(columns={'Organization: Organization Name': 'Organization Name'})
    df_four_level = df_four_level[df_four_level['Program'].notna() &
                                  df_four_level['Strategy'].notna() &
                                  df_four_level['Organization Name'].notna() &
                                  df_four_level['Amount'].notna()]

    df_four_level['Strategy'] = df_four_level['Strategy'].fillna('unspecified')
    df_four_level['Substrategy'] = df_four_level['Substrategy'].fillna('unspecified')

    df_four_grouped = df_four_level.groupby(['Program', 'Strategy', 'Substrategy', 'Organization Name'],
                                            as_index=False)['Amount'].sum()

    print(f"filtering organizations with amounts less than ${MIN_ORG_AMOUNT:,.0f}...")
    df_four_final = calculate_angles_and_filter_four_level(df_four_grouped, min_amount=MIN_ORG_AMOUNT)

    df_four_final['Is_Other'] = df_four_final['Organization Name'].str.startswith('Other').astype(int)
    df_four_final = df_four_final.sort_values(['Program', 'Strategy', 'Substrategy', 'Is_Other', 'Amount'],
                                              ascending=[True, True, True, True, False])
    df_four_final = df_four_final.drop('Is_Other', axis=1)

    ids = []
    labels = []
    parents = []
    values = []
    colors = []

    for program in df_four_final['Program'].unique():
        ids.append(program)
        labels.append(program)
        parents.append('')
        values.append(0)
        colors.append(program_colors.get(program, '#CCCCCC'))

    for (program, strategy) in df_four_final[['Program', 'Strategy']].drop_duplicates().values:
        strategy_id = f"{program}_{strategy}"
        ids.append(strategy_id)
        labels.append('' if strategy.lower() == 'unspecified' else strategy)
        parents.append(program)
        values.append(0)
        colors.append(lighten_color(program_colors.get(program, '#CCCCCC'), factor=0.25))

    for (program, strategy, substrategy) in df_four_final[
        ['Program', 'Strategy', 'Substrategy']].drop_duplicates().values:
        strategy_id = f"{program}_{strategy}"
        substrategy_id = f"{program}_{strategy}_{substrategy}"
        ids.append(substrategy_id)
        labels.append('' if substrategy.lower() == 'unspecified' else substrategy)
        parents.append(strategy_id)
        values.append(0)
        colors.append(lighten_color(program_colors.get(program, '#CCCCCC'), factor=0.5))

    for idx, row in df_four_final.iterrows():
        program = row['Program']
        strategy = row['Strategy']
        substrategy = row['Substrategy']
        org = row['Organization Name']
        amount = row['Amount']

        org_id = f"{program}_{strategy}_{substrategy}_{org}_{idx}"
        substrategy_id = f"{program}_{strategy}_{substrategy}"

        ids.append(org_id)
        labels.append(org)
        parents.append(substrategy_id)
        values.append(amount)

        if org.startswith('Other'):
            colors.append('#CCCCCC')
        else:
            colors.append(lighten_color(program_colors.get(program, '#CCCCCC'), factor=0.7))

    fig = go.Figure(go.Sunburst(
        ids=ids,
        labels=labels,
        parents=parents,
        values=values,
        marker=dict(
            colors=colors,
            line=dict(width=2, color='white')
        ),
        insidetextorientation='radial',
        hovertemplate='<b>%{label}</b><br>Amount: %{value:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title=f'Grants by Program, Strategy, Substrategy, and Organizations (min ${MIN_ORG_AMOUNT / 1000000:.0f}M per org)',
        width=1600,
        height=1600,
        font_size=10
    )

    output_path = 'outputs/sunburst_four_level.png'
    fig.write_image(output_path)
    print(f"four-level sunburst chart saved to {output_path}")

    output_path_svg = 'outputs/sunburst_four_level.svg'
    fig.write_image(output_path_svg)
    print(f"four-level sunburst chart svg saved to {output_path_svg}")

    output_path_html = 'outputs/sunburst_four_level.html'
    fig.write_html(output_path_html)
    print(f"interactive four-level sunburst chart saved to {output_path_html}")

    original_count = len(df_four_grouped)
    filtered_count = len(df_four_final[~df_four_final['Organization Name'].str.startswith('Other')])
    print(f"organizations: {original_count} original, {filtered_count} after filtering")

# treemap by program and top 10 grantees per program
if GENERATE_TREEMAP_TOP_GRANTEES:
    print("generating treemap by top grantees per program...")

    df_grantee = df[['Program', 'Organization: Organization Name', 'Amount']].copy()
    df_grantee = df_grantee.rename(columns={'Organization: Organization Name': 'Organization Name'})
    df_grantee = df_grantee[df_grantee['Program'].notna() &
                            df_grantee['Organization Name'].notna() &
                            df_grantee['Amount'].notna()]

    df_grantee_grouped = df_grantee.groupby(['Program', 'Organization Name'], as_index=False)['Amount'].sum()

    result_rows = []
    for program in df_grantee_grouped['Program'].unique():
        program_data = df_grantee_grouped[df_grantee_grouped['Program'] == program].copy()
        program_data = program_data.sort_values('Amount', ascending=False)

        top_10_program = program_data.head(10)
        result_rows.append(top_10_program)

        if len(program_data) > 10:
            other_amount = program_data.iloc[10:]['Amount'].sum()
            other_count = len(program_data) - 10
            other_row = pd.DataFrame({
                'Program': [program],
                'Organization Name': [f'Other ({other_count})'],
                'Amount': [other_amount]
            })
            result_rows.append(other_row)

    df_grantee_grouped = pd.concat(result_rows, ignore_index=True)

    total_top_grantees = df_grantee_grouped['Amount'].sum()

    labels = []
    parents = []
    values = []
    colors = []
    custom_text_static = []
    custom_text_interactive = []

    for program in df_grantee_grouped['Program'].unique():
        labels.append(program)
        parents.append('')
        values.append(0)
        colors.append(program_colors.get(program, '#CCCCCC'))
        custom_text_static.append(program)
        custom_text_interactive.append(program)

    for idx, row in df_grantee_grouped.iterrows():
        program = row['Program']
        grantee = row['Organization Name']
        amount = row['Amount']

        unique_label = f"{program}_{grantee}_{idx}"
        labels.append(unique_label)
        parents.append(program)
        values.append(amount)

        if grantee.startswith('Other'):
            colors.append('#CCCCCC')
        else:
            colors.append(lighten_color(program_colors.get(program, '#CCCCCC')))

        if amount / total_top_grantees < 0.01:
            custom_text_static.append('')
        else:
            custom_text_static.append(wrap_text(grantee, width=25))

        custom_text_interactive.append(wrap_text(grantee, width=25))

    fig_static = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        text=custom_text_static,
        textinfo='text',
        marker=dict(
            colors=colors,
            line=dict(width=2, color='white')
        ),
        textposition='middle center',
        textfont=dict(size=12),
        hovertemplate='<b>%{text}</b><br>Amount: %{value:,.0f}<extra></extra>'
    ))

    fig_static.update_layout(
        title='Grants by Program and Top 10 Grantees per Program',
        width=1600,
        height=1000,
        font_size=11
    )

    output_path = 'outputs/treemap_program_top_grantees.png'
    fig_static.write_image(output_path)
    print(f"treemap with top grantees per program saved to {output_path}")

    fig_interactive = go.Figure(go.Treemap(
        labels=labels,
        parents=parents,
        values=values,
        text=custom_text_interactive,
        textinfo='text',
        marker=dict(
            colors=colors,
            line=dict(width=2, color='white')
        ),
        textposition='middle center',
        textfont=dict(size=12),
        hovertemplate='<b>%{text}</b><br>Amount: %{value:,.0f}<extra></extra>'
    ))

    fig_interactive.update_layout(
        title='Grants by Program and Top 10 Grantees per Program',
        width=1600,
        height=1000,
        font_size=11
    )

    output_path_html = 'outputs/treemap_program_top_grantees.html'
    fig_interactive.write_html(output_path_html)
    print(f"interactive treemap saved to {output_path_html}")

# csv pivot table
if GENERATE_CSV:
    print("generating csv pivot table...")

    output_path = 'outputs/pivot_program_strategy.csv'
    df_grouped.to_csv(output_path, index=False)
    print(f"csv pivot table saved to {output_path}")

    print("generating program-grantee pivot table...")
    df_grantee_all = df[['Program', 'Organization: Organization Name', 'Amount']].copy()
    df_grantee_all = df_grantee_all.rename(columns={'Organization: Organization Name': 'Organization Name'})
    df_grantee_all = df_grantee_all[df_grantee_all['Program'].notna() &
                                    df_grantee_all['Organization Name'].notna() &
                                    df_grantee_all['Amount'].notna()]
    df_grantee_all_grouped = df_grantee_all.groupby(['Program', 'Organization Name'], as_index=False)['Amount'].sum()
    df_grantee_all_grouped = df_grantee_all_grouped.sort_values(['Program', 'Amount'], ascending=[True, False])
    output_path = 'outputs/pivot_program_grantee.csv'
    df_grantee_all_grouped.to_csv(output_path, index=False)
    print(f"program-grantee pivot table saved to {output_path}")

# summary
print(f"\ntotal amount visualized: ${total_amount:,.0f}")
print(f"number of program-strategy combinations: {len(df_grouped)}")