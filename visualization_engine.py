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

# three-level sunburst visualization (program > strategy > top 5 orgs per strategy)
if GENERATE_SUNBURST_THREE_LEVEL:
    print("generating three-level sunburst (program > strategy > top 5 orgs)...")

    # prepare data with organization information
    df_three_level = df[['Program', 'Strategy', 'Organization: Organization Name', 'Amount']].copy()
    df_three_level = df_three_level.rename(columns={'Organization: Organization Name': 'Organization Name'})
    df_three_level = df_three_level[df_three_level['Program'].notna() &
                                    df_three_level['Strategy'].notna() &
                                    df_three_level['Organization Name'].notna() &
                                    df_three_level['Amount'].notna()]

    # replace null/blank strategies with "unspecified"
    df_three_level['Strategy'] = df_three_level['Strategy'].fillna('unspecified')

    # group by program, strategy, and organization
    df_three_grouped = df_three_level.groupby(['Program', 'Strategy', 'Organization Name'], as_index=False)[
        'Amount'].sum()

    # for each program-strategy combination, get top 5 organizations only
    result_rows = []
    for (program, strategy) in df_three_grouped[['Program', 'Strategy']].drop_duplicates().values:
        strategy_data = df_three_grouped[(df_three_grouped['Program'] == program) &
                                         (df_three_grouped['Strategy'] == strategy)].copy()
        strategy_data = strategy_data.sort_values('Amount', ascending=False)

        # get top 5 for this strategy
        top_5_strategy = strategy_data.head(5)
        result_rows.append(top_5_strategy)

    # combine all results
    df_three_final = pd.concat(result_rows, ignore_index=True)

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
        parents.append('')  # no parent for top level
        values.append(0)  # set to 0 - plotly will calculate from children
        colors.append(program_colors.get(program, '#CCCCCC'))

    # level 2: add strategy level (middle ring)
    for (program, strategy) in df_three_final[['Program', 'Strategy']].drop_duplicates().values:
        # create unique id for strategy to avoid duplicates across programs
        strategy_id = f"{program}_{strategy}"
        ids.append(strategy_id)
        # hide label for unspecified strategies
        labels.append('' if strategy.lower() == 'unspecified' else strategy)
        parents.append(program)
        values.append(0)  # set to 0 - plotly will calculate from children
        # lighten program color for strategies
        colors.append(lighten_color(program_colors.get(program, '#CCCCCC'), factor=0.3))

    # level 3: add organization level (outermost ring)
    for idx, row in df_three_final.iterrows():
        program = row['Program']
        strategy = row['Strategy']
        org = row['Organization Name']
        amount = row['Amount']

        # create unique id for organization
        org_id = f"{program}_{strategy}_{org}_{idx}"
        strategy_id = f"{program}_{strategy}"

        ids.append(org_id)
        labels.append(org)  # display only organization name
        # parent is the unique strategy id
        parents.append(strategy_id)
        values.append(amount)

        # use more lightened program color for organizations
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
        title='Grants by Program, Strategy, and Top 5 Organizations per Strategy',
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

# four-level sunburst visualization (program > strategy > substrategy > top 5 orgs)
if GENERATE_SUNBURST_FOUR_LEVEL:
    print("generating four-level sunburst (program > strategy > substrategy > top 5 orgs)...")

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

    # for each program-strategy-substrategy combination, get top 5 organizations only
    result_rows = []
    for (program, strategy, substrategy) in df_four_grouped[['Program', 'Strategy', 'Substrategy']].drop_duplicates().values:
        substrategy_data = df_four_grouped[(df_four_grouped['Program'] == program) &
                                           (df_four_grouped['Strategy'] == strategy) &
                                           (df_four_grouped['Substrategy'] == substrategy)].copy()
        substrategy_data = substrategy_data.sort_values('Amount', ascending=False)

        # get top 5 for this substrategy
        top_5_substrategy = substrategy_data.head(5)
        result_rows.append(top_5_substrategy)

        # if there are more than 5 organizations, lump the rest into "other"
        if len(substrategy_data) > 5:
            other_amount = substrategy_data.iloc[5:]['Amount'].sum()
            other_count = len(substrategy_data) - 5
            other_row = pd.DataFrame({
                'Program': [program],
                'Strategy': [strategy],
                'Substrategy': [substrategy],
                'Organization Name': [f'Other ({other_count})'],
                'Amount': [other_amount]
            })
            result_rows.append(other_row)

    # combine all results
    df_four_final = pd.concat(result_rows, ignore_index=True)

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
    for (program, strategy, substrategy) in df_four_final[['Program', 'Strategy', 'Substrategy']].drop_duplicates().values:
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
        title='Grants by Program, Strategy, Substrategy, and Top 5 Organizations',
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
            other_row = pd.DataFrame({
                'Program': [program],
                'Organization Name': ['Other'],
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
        if grantee == 'Other':
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