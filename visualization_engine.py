import pandas as pd
import plotly.graph_objects as go
import os
import textwrap

# visualization settings
GENERATE_TREEMAP = True
GENERATE_SUNBURST = True
GENERATE_CSV = True

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

# sunburst visualization
if GENERATE_SUNBURST:
    print("generating sunburst...")

    # create lists for sunburst
    labels = []
    parents = []
    values = []
    colors = []

    # calculate program totals
    program_totals = df_grouped.groupby('Program')['Amount'].sum().to_dict()

    # first, add program level (parents)
    for program in df_grouped['Program'].unique():
        labels.append(program)
        parents.append('')  # no parent for top level
        values.append(program_totals[program])
        colors.append(program_colors.get(program, '#CCCCCC'))

    # then, add strategy level (children)
    for idx, row in df_grouped.iterrows():
        program = row['Program']
        strategy = row['Strategy']
        amount = row['Amount']

        labels.append(strategy)
        parents.append(program)
        values.append(amount)
        # use lightened version of program color for strategies
        colors.append(lighten_color(program_colors.get(program, '#CCCCCC')))

    # create the sunburst chart
    fig = go.Figure(go.Sunburst(
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
        title='Grants by Program and Strategy',
        width=1200,
        height=1200,
        font_size=12
    )

    # save as static image
    output_path = 'outputs/sunburst_program_strategy.png'
    fig.write_image(output_path)
    print(f"sunburst chart saved to {output_path}")

# csv pivot table
if GENERATE_CSV:
    print("generating csv pivot table...")

    # export the grouped data as csv
    output_path = 'outputs/pivot_program_strategy.csv'
    df_grouped.to_csv(output_path, index=False)
    print(f"csv pivot table saved to {output_path}")

# summary
print(f"\ntotal amount visualized: ${total_amount:,.0f}")
print(f"number of program-strategy combinations: {len(df_grouped)}")