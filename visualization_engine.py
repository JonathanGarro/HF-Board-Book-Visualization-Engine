import pandas as pd
import plotly.graph_objects as go
import os
import textwrap

# visualization settings
GENERATE_TREEMAP = True
GENERATE_SUNBURST = True
GENERATE_SUNBURST_THREE_LEVEL = True
GENERATE_SUNBURST_FOUR_LEVEL = True
GENERATE_CSV = True
GENERATE_TREEMAP_TOP_GRANTEES = True

# threshold setting for angle-based visualizations (minimum angle in degrees)
MIN_ANGLE_DEGREES = 10

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
    # wrap text and join with html line breaks
    wrapped = textwrap.fill(text, width=width)
    return wrapped.replace('\n', '<br>')


def lighten_color(hex_color, factor=0.4):
    """lighten a hex color by mixing it with white"""
    # remove the # if present
    hex_color = hex_color.lstrip('#')
    # convert to rgb
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    # lighten by moving towards white (255, 255, 255)
    r = int(r + (255 - r) * factor)
    g = int(g + (255 - g) * factor)
    b = int(b + (255 - b) * factor)
    # convert back to hex
    return f'#{r:02x}{g:02x}{b:02x}'


def calculate_angles_and_filter(df_grouped, min_angle=10):
    """
    calculate the angle each organization would occupy when its parent strategy
    is fully visible (360 degrees) and filter out organizations below threshold
    """
    # group by program and strategy to get strategy totals
    strategy_totals = df_grouped.groupby(['Program', 'Strategy'])['Amount'].sum().reset_index()
    strategy_totals.columns = ['Program', 'Strategy', 'Strategy_Total']

    # merge strategy totals back to get each org's percentage of its strategy
    df_with_totals = df_grouped.merge(strategy_totals, on=['Program', 'Strategy'])

    # calculate the angle each org would occupy if strategy was the full circle (360°)
    df_with_totals['Angle_In_Strategy'] = (df_with_totals['Amount'] /
                                           df_with_totals['Strategy_Total']) * 360

    # filter organizations that would be too small
    df_filtered = df_with_totals[df_with_totals['Angle_In_Strategy'] >= min_angle].copy()

    # for each strategy, collect filtered-out orgs into "other"
    result_rows = []
    for (program, strategy) in df_with_totals[['Program', 'Strategy']].drop_duplicates().values:
        strategy_data = df_with_totals[(df_with_totals['Program'] == program) &
                                       (df_with_totals['Strategy'] == strategy)]

        kept_orgs = df_filtered[(df_filtered['Program'] == program) &
                                (df_filtered['Strategy'] == strategy)]
        result_rows.append(kept_orgs[['Program', 'Strategy', 'Organization Name', 'Amount']])

        # if there are orgs filtered out, create "other" category
        filtered_out = strategy_data[strategy_data['Angle_In_Strategy'] < min_angle]
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

    return pd.concat(result_rows, ignore_index=True)


def calculate_angles_and_filter_four_level(df_grouped, min_angle=10):
    """
    calculate the angle each organization would occupy when its parent substrategy
    is fully visible (360 degrees) and filter out organizations below threshold
    """
    # group by program, strategy, and substrategy to get substrategy totals
    substrategy_totals = df_grouped.groupby(['Program', 'Strategy', 'Substrategy'])['Amount'].sum().reset_index()
    substrategy_totals.columns = ['Program', 'Strategy', 'Substrategy', 'Substrategy_Total']

    # merge substrategy totals back to get each org's percentage of its substrategy
    df_with_totals = df_grouped.merge(substrategy_totals, on=['Program', 'Strategy', 'Substrategy'])

    # calculate the angle each org would occupy if substrategy was the full circle (360°)
    df_with_totals['Angle_In_Substrategy'] = (df_with_totals['Amount'] /
                                              df_with_totals['Substrategy_Total']) * 360

    # filter organizations that would be too small
    df_filtered = df_with_totals[df_with_totals['Angle_In_Substrategy'] >= min_angle].copy()

    # for each substrategy, collect filtered-out orgs into "other"
    result_rows = []
    for (program, strategy, substrategy) in df_with_totals[
        ['Program', 'Strategy', 'Substrategy']].drop_duplicates().values:
        substrategy_data = df_with_totals[(df_with_totals['Program'] == program) &
                                          (df_with_totals['Strategy'] == strategy) &
                                          (df_with_totals['Substrategy'] == substrategy)]

        kept_orgs = df_filtered[(df_filtered['Program'] == program) &
                                (df_filtered['Strategy'] == strategy) &
                                (df_filtered['Substrategy'] == substrategy)]
        result_rows.append(kept_orgs[['Program', 'Strategy', 'Substrategy', 'Organization Name', 'Amount']])

        # if there are orgs filtered out, create "other" category
        filtered_out = substrategy_data[substrategy_data['Angle_In_Substrategy'] < min_angle]
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

    return pd.concat(result_rows, ignore_index=True)


# data loading
df = pd.read_csv('00OUf000008PyafMAC_mapped.csv')

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

    # create lists for treemap
    labels = []
    parents = []
    values = []
    colors = []
    custom_text = []

    # first, add program level (parents)
    for program in df_grouped['Program'].unique():
        labels.append(program)
        parents.append('')  # no parent for top level
        values.append(0)  # will be calculated by plotly
        colors.append(program_colors.get(program, '#CCCCCC'))
        custom_text.append(program)

    # then, add strategy level (children)
    for idx, row in df_grouped.iterrows():
        program = row['Program']
        strategy = row['Strategy']
        amount = row['Amount']

        # create unique label by combining program and strategy to avoid duplicates
        unique_label = f"{program}_{strategy}_{idx}"
        labels.append(unique_label)
        parents.append(program)
        values.append(amount)
        # use lightened version of program color for strategies
        colors.append(lighten_color(program_colors.get(program, '#CCCCCC')))

        # hide text for very small boxes (less than 1% of total) or unspecified strategies
        if amount / total_amount < 0.01 or strategy == 'unspecified':
            custom_text.append('')  # empty string hides the label
        else:
            custom_text.append(wrap_text(strategy))

    # create the treemap
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

    # save as static image
    output_path = 'outputs/treemap_program_strategy.png'
    fig.write_image(output_path)
    print(f"treemap saved to {output_path}")

    # save as interactive html
    output_path_html = 'outputs/treemap_program_strategy.html'
    fig.write_html(output_path_html)
    print(f"interactive treemap saved to {output_path_html}")

# sunburst visualization
if GENERATE_SUNBURST:
    print("generating sunburst...")

    # create lists for sunburst
    ids = []
    labels = []
    parents = []
    values = []
    colors = []

    # first, add program level (parents)
    for program in df_grouped['Program'].unique():
        ids.append(program)
        labels.append(program)
        parents.append('')  # no parent for top level
        values.append(0)  # set to 0 - plotly will calculate from children
        colors.append(program_colors.get(program, '#CCCCCC'))

    # then, add strategy level (children)
    for idx, row in df_grouped.iterrows():
        program = row['Program']
        strategy = row['Strategy']
        amount = row['Amount']

        # create unique id for strategy
        strategy_id = f"{program}_{strategy}_{idx}"
        ids.append(strategy_id)
        # hide label for unspecified strategies
        labels.append('' if strategy.lower() == 'unspecified' else strategy)
        parents.append(program)
        values.append(amount)
        # use lightened version of program color for strategies
        colors.append(lighten_color(program_colors.get(program, '#CCCCCC')))

    # create the sunburst chart
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
        textfont=dict(size=12),
        hovertemplate='<b>%{label}</b><br>Amount: %{value:,.0f}<extra></extra>'
    ))

    fig.update_layout(
        title='Grants by Program and Strategy',
        width=1200,
        height=1200,
        font_size=12
    )

    # save as static image
    output_path = 'outputs/sunburst_program_strategy.png'
    fig.write_image(output_path)
    print(f"sunburst chart saved to {output_path}")

    # save as svg for editing in illustrator
    output_path_svg = 'outputs/sunburst_program_strategy.svg'
    fig.write_image(output_path_svg)
    print(f"sunburst chart svg saved to {output_path_svg}")

    # save as interactive html
    output_path_html = 'outputs/sunburst_program_strategy.html'
    fig.write_html(output_path_html)
    print(f"interactive sunburst chart saved to {output_path_html}")

# complete three-level sunburst section with angle filtering
if GENERATE_SUNBURST_THREE_LEVEL:
    print("generating three-level sunburst (program > strategy > top orgs with angle filtering)...")

    # prepare data with organization information
    df_three_level = df[['Program', 'Strategy', 'Organization: Organization Name', 'Amount']].copy()
    df_three_level = df_three_level.rename(columns={'Organization: Organization Name': 'Organization Name'})

    # replace null/blank strategies with "unspecified" BEFORE filtering
    df_three_level['Strategy'] = df_three_level['Strategy'].fillna('unspecified')

    # now filter - keep rows with program, org name, and amount (strategy already filled)
    df_three_level = df_three_level[df_three_level['Program'].notna() &
                                    df_three_level['Organization Name'].notna() &
                                    df_three_level['Amount'].notna()]

    # group by program, strategy, and organization
    df_three_grouped = df_three_level.groupby(['Program', 'Strategy', 'Organization Name'],
                                              as_index=False)['Amount'].sum()

    # apply angle-based filtering
    print(f"filtering organizations with angles less than {MIN_ANGLE_DEGREES} degrees when zoomed...")
    df_three_final = calculate_angles_and_filter(df_three_grouped, min_angle=MIN_ANGLE_DEGREES)

    # sort so "other" slices appear at the end of each strategy group
    df_three_final['Is_Other'] = df_three_final['Organization Name'].str.startswith('Other').astype(int)
    df_three_final = df_three_final.sort_values(['Program', 'Strategy', 'Is_Other', 'Amount'],
                                                ascending=[True, True, True, False])
    df_three_final = df_three_final.drop('Is_Other', axis=1)

    # create lists for sunburst
    ids = []
    labels = []
    parents = []
    values = []
    colors = []

    # level 1: add program level (innermost ring)
    for program in df_three_final['Program'].unique():
        ids.append(program)
        labels.append(program)
        parents.append('')
        values.append(0)
        colors.append(program_colors.get(program, '#CCCCCC'))

    # level 2: add strategy level (middle ring)
    for (program, strategy) in df_three_final[['Program', 'Strategy']].drop_duplicates().values:
        strategy_id = f"{program}_{strategy}"
        ids.append(strategy_id)
        labels.append('' if strategy.lower() == 'unspecified' else strategy)
        parents.append(program)
        values.append(0)
        colors.append(lighten_color(program_colors.get(program, '#CCCCCC'), factor=0.3))

    # level 3: add organization level (outermost ring)
    for idx, row in df_three_final.iterrows():
        program = row['Program']
        strategy = row['Strategy']
        org = row['Organization Name']
        amount = row['Amount']

        org_id = f"{program}_{strategy}_{org}_{idx}"
        strategy_id = f"{program}_{strategy}"

        ids.append(org_id)
        labels.append(org)
        parents.append(strategy_id)
        values.append(amount)

        if org.startswith('Other'):
            colors.append('#CCCCCC')
        else:
            colors.append(lighten_color(program_colors.get(program, '#CCCCCC'), factor=0.6))

    # create the sunburst chart
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
        title=f'Grants by Program, Strategy, and Organizations (min {MIN_ANGLE_DEGREES}° when zoomed)',
        width=1400,
        height=1400,
        font_size=11
    )

    # save as static image
    output_path = 'outputs/sunburst_three_level.png'
    fig.write_image(output_path)
    print(f"three-level sunburst chart saved to {output_path}")

    # save as interactive html
    output_path_html = 'outputs/sunburst_three_level.html'
    fig.write_html(output_path_html)
    print(f"interactive three-level sunburst chart saved to {output_path_html}")

    # print stats about filtering
    original_count = len(df_three_grouped)
    filtered_count = len(df_three_final[~df_three_final['Organization Name'].str.startswith('Other')])
    print(f"organizations: {original_count} original, {filtered_count} after filtering")

    # create directory for individual program sunbursts
    program_dir = 'outputs/sunburst_three_level_programs'
    os.makedirs(program_dir, exist_ok=True)
    print(f"\ngenerating zoomed sunbursts for each program...")

    # generate a zoomed sunburst for each program
    for program in df_three_final['Program'].unique():
        # filter data for this program only
        program_data = df_three_final[df_three_final['Program'] == program]

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

        # level 1: add strategy level (child of hidden root)
        for strategy in program_data['Strategy'].unique():
            strategy_id = f"{program}_{strategy}"
            prog_ids.append(strategy_id)
            prog_labels.append('' if strategy.lower() == 'unspecified' else strategy)
            prog_parents.append('root')  # strategies point to hidden root
            prog_values.append(0)
            prog_colors.append(lighten_color(program_colors.get(program, '#CCCCCC'), factor=0.3))

        # level 2: add organization level
        for idx, row in program_data.iterrows():
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

        # create the sunburst chart for this program
        fig_program = go.Figure(go.Sunburst(
            ids=prog_ids,
            labels=prog_labels,
            parents=prog_parents,
            values=prog_values,
            marker=dict(
                colors=prog_colors,
                line=dict(width=2, color='white')
            ),
            insidetextorientation='radial',
            hovertemplate='<b>%{label}</b><br>Amount: %{value:,.0f}<extra></extra>'
        ))

        fig_program.update_layout(
            title=f'{program} - Strategies and Organizations',
            width=1400,
            height=1400,
            font_size=11
        )

        # create clean filename from program name
        program_filename = program.replace(' ', '_').replace('&', 'and').replace(':', '').lower()

        # save as png
        output_path_png = f'{program_dir}/{program_filename}_zoomed.png'
        fig_program.write_image(output_path_png)

        # save as svg for editing in illustrator
        output_path_svg = f'{program_dir}/{program_filename}_zoomed.svg'
        fig_program.write_image(output_path_svg)

        print(f"saved zoomed sunburst for {program}")

# four-level sunburst visualization (program > strategy > substrategy > top orgs with angle filtering)
if GENERATE_SUNBURST_FOUR_LEVEL:
    print("generating four-level sunburst (program > strategy > substrategy > top orgs with angle filtering)...")

    # prepare data with organization information
    df_four_level = df[['Program', 'Strategy', 'Substrategy', 'Organization: Organization Name', 'Amount']].copy()
    df_four_level = df_four_level.rename(columns={'Organization: Organization Name': 'Organization Name'})
    df_four_level = df_four_level[df_four_level['Program'].notna() &
                                  df_four_level['Strategy'].notna() &
                                  df_four_level['Organization Name'].notna() &
                                  df_four_level['Amount'].notna()]

    # replace null/blank strategies and substrategies with "unspecified"
    df_four_level['Strategy'] = df_four_level['Strategy'].fillna('unspecified')
    df_four_level['Substrategy'] = df_four_level['Substrategy'].fillna('unspecified')

    # group by program, strategy, substrategy, and organization
    df_four_grouped = df_four_level.groupby(['Program', 'Strategy', 'Substrategy', 'Organization Name'],
                                            as_index=False)['Amount'].sum()

    # apply angle-based filtering
    print(f"filtering organizations with angles less than {MIN_ANGLE_DEGREES} degrees when zoomed...")
    df_four_final = calculate_angles_and_filter_four_level(df_four_grouped, min_angle=MIN_ANGLE_DEGREES)

    # sort so "Other" slices appear at the end of each substrategy group
    df_four_final['Is_Other'] = df_four_final['Organization Name'].str.startswith('Other').astype(int)
    df_four_final = df_four_final.sort_values(['Program', 'Strategy', 'Substrategy', 'Is_Other', 'Amount'],
                                              ascending=[True, True, True, True, False])
    df_four_final = df_four_final.drop('Is_Other', axis=1)

    # create lists for sunburst
    ids = []
    labels = []
    parents = []
    values = []
    colors = []

    # level 1: add program level (innermost ring)
    for program in df_four_final['Program'].unique():
        ids.append(program)
        labels.append(program)
        parents.append('')  # no parent for top level
        values.append(0)  # set to 0 - plotly will calculate from children
        colors.append(program_colors.get(program, '#CCCCCC'))

    # level 2: add strategy level
    for (program, strategy) in df_four_final[['Program', 'Strategy']].drop_duplicates().values:
        # create unique id for strategy to avoid duplicates across programs
        strategy_id = f"{program}_{strategy}"
        ids.append(strategy_id)
        # hide label for unspecified strategies
        labels.append('' if strategy.lower() == 'unspecified' else strategy)
        parents.append(program)
        values.append(0)  # set to 0 - plotly will calculate from children
        # lighten program color for strategies
        colors.append(lighten_color(program_colors.get(program, '#CCCCCC'), factor=0.25))

    # level 3: add substrategy level
    for (program, strategy, substrategy) in df_four_final[
        ['Program', 'Strategy', 'Substrategy']].drop_duplicates().values:
        # create unique id for substrategy
        strategy_id = f"{program}_{strategy}"
        substrategy_id = f"{program}_{strategy}_{substrategy}"
        ids.append(substrategy_id)
        # hide label for unspecified substrategies
        labels.append('' if substrategy.lower() == 'unspecified' else substrategy)
        parents.append(strategy_id)
        values.append(0)  # set to 0 - plotly will calculate from children
        # further lighten program color for substrategies
        colors.append(lighten_color(program_colors.get(program, '#CCCCCC'), factor=0.5))

    # level 4: add organization level (outermost ring)
    for idx, row in df_four_final.iterrows():
        program = row['Program']
        strategy = row['Strategy']
        substrategy = row['Substrategy']
        org = row['Organization Name']
        amount = row['Amount']

        # create unique id for organization
        org_id = f"{program}_{strategy}_{substrategy}_{org}_{idx}"
        substrategy_id = f"{program}_{strategy}_{substrategy}"

        ids.append(org_id)
        labels.append(org)  # display only organization name
        # parent is the unique substrategy id
        parents.append(substrategy_id)
        values.append(amount)

        # use grey for "other", otherwise use lightened program color
        if org.startswith('Other'):
            colors.append('#CCCCCC')
        else:
            colors.append(lighten_color(program_colors.get(program, '#CCCCCC'), factor=0.7))

    # create the sunburst chart
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
        title=f'Grants by Program, Strategy, Substrategy, and Organizations (min {MIN_ANGLE_DEGREES}° when zoomed)',
        width=1600,
        height=1600,
        font_size=10
    )

    # save as static image
    output_path = 'outputs/sunburst_four_level.png'
    fig.write_image(output_path)
    print(f"four-level sunburst chart saved to {output_path}")

    # save as svg for editing in illustrator
    output_path_svg = 'outputs/sunburst_four_level.svg'
    fig.write_image(output_path_svg)
    print(f"four-level sunburst chart svg saved to {output_path_svg}")

    # save as interactive html
    output_path_html = 'outputs/sunburst_four_level.html'
    fig.write_html(output_path_html)
    print(f"interactive four-level sunburst chart saved to {output_path_html}")

    # print stats about filtering
    original_count = len(df_four_grouped)
    filtered_count = len(df_four_final[~df_four_final['Organization Name'].str.startswith('Other')])
    print(f"organizations: {original_count} original, {filtered_count} after filtering")

# treemap by program and top 10 grantees per program
if GENERATE_TREEMAP_TOP_GRANTEES:
    print("generating treemap by top grantees per program...")

    # prepare data with grantee information - note the column name has a colon
    df_grantee = df[['Program', 'Organization: Organization Name', 'Amount']].copy()
    df_grantee = df_grantee.rename(columns={'Organization: Organization Name': 'Organization Name'})
    df_grantee = df_grantee[df_grantee['Program'].notna() &
                            df_grantee['Organization Name'].notna() &
                            df_grantee['Amount'].notna()]

    # group by program and grantee
    df_grantee_grouped = df_grantee.groupby(['Program', 'Organization Name'], as_index=False)['Amount'].sum()

    # for each program, identify top 10 grantees and lump rest into "other"
    result_rows = []
    for program in df_grantee_grouped['Program'].unique():
        program_data = df_grantee_grouped[df_grantee_grouped['Program'] == program].copy()
        program_data = program_data.sort_values('Amount', ascending=False)

        # get top 10 for this program
        top_10_program = program_data.head(10)
        result_rows.append(top_10_program)

        # if there are more than 10 grantees, lump the rest into "other"
        if len(program_data) > 10:
            other_amount = program_data.iloc[10:]['Amount'].sum()
            other_count = len(program_data) - 10
            other_row = pd.DataFrame({
                'Program': [program],
                'Organization Name': [f'Other ({other_count})'],
                'Amount': [other_amount]
            })
            result_rows.append(other_row)

    # combine all results
    df_grantee_grouped = pd.concat(result_rows, ignore_index=True)

    # calculate total for this visualization
    total_top_grantees = df_grantee_grouped['Amount'].sum()

    # create lists for treemap
    labels = []
    parents = []
    values = []
    colors = []
    custom_text_static = []
    custom_text_interactive = []

    # first, add program level (parents)
    for program in df_grantee_grouped['Program'].unique():
        labels.append(program)
        parents.append('')  # no parent for top level
        values.append(0)  # will be calculated by plotly
        colors.append(program_colors.get(program, '#CCCCCC'))
        custom_text_static.append(program)
        custom_text_interactive.append(program)

    # then, add grantee level (children)
    for idx, row in df_grantee_grouped.iterrows():
        program = row['Program']
        grantee = row['Organization Name']
        amount = row['Amount']

        # create unique label
        unique_label = f"{program}_{grantee}_{idx}"
        labels.append(unique_label)
        parents.append(program)
        values.append(amount)

        # use grey for "other", lightened program color for specific grantees
        if grantee.startswith('Other'):
            colors.append('#CCCCCC')
        else:
            colors.append(lighten_color(program_colors.get(program, '#CCCCCC')))

        # for static: hide text for very small boxes (less than 1% of total)
        if amount / total_top_grantees < 0.01:
            custom_text_static.append('')
        else:
            custom_text_static.append(wrap_text(grantee, width=25))

        # for interactive: always show text (will be visible when zoomed)
        custom_text_interactive.append(wrap_text(grantee, width=25))

    # create the static treemap
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

    # save as static image
    output_path = 'outputs/treemap_program_top_grantees.png'
    fig_static.write_image(output_path)
    print(f"treemap with top grantees per program saved to {output_path}")

    # create the interactive treemap (with all labels)
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

    # save as interactive html
    output_path_html = 'outputs/treemap_program_top_grantees.html'
    fig_interactive.write_html(output_path_html)
    print(f"interactive treemap saved to {output_path_html}")

# csv pivot table
if GENERATE_CSV:
    print("generating csv pivot table...")

    # export the grouped data as csv
    output_path = 'outputs/pivot_program_strategy.csv'
    df_grouped.to_csv(output_path, index=False)
    print(f"csv pivot table saved to {output_path}")

    # export program > grantee pivot
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