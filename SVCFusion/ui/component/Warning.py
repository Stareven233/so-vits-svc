def WarningHTML(message):
    return f"""
<div style="
    display: flex;
    align-items: center;
    color: #ad6800;
    background: #fffbe6;
    border: 1.5px solid #faad14;
    padding: 14px 16px;
    border-radius: 6px;
    font-size: 16px;
    font-weight: 500;
    margin: 12px 0px !important;
">
    <span style="font-size: 22px; margin-right: 12px;">&#9888;&#65039;</span>
    <span style="color: black !important">{message}</span>
</div>
"""
