"""Colores y estilo comunes de las gráficas del proyecto.

Paleta categórica validada para daltonismo (azul y naranja) sobre fondo claro.
"""

COLOR_TIENDA = "#2a78d6"
COLOR_PRODUCTO = "#eb6834"
COLOR_TEXTO = "#0b0b0b"
COLOR_TEXTO_SECUNDARIO = "#52514e"
COLOR_FONDO = "#fcfcfb"
COLOR_EJE = "#d8d7d2"
COLOR_GRILLA = "#ecebe7"


def estilo_ejes(ax, grilla: str = "y") -> None:
    """Fondo claro, sin bordes superiores ni laterales y grilla tenue."""
    ax.set_facecolor(COLOR_FONDO)
    ax.tick_params(colors=COLOR_TEXTO_SECUNDARIO, length=0)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color(COLOR_EJE)
    ax.grid(axis=grilla, color=COLOR_GRILLA, linewidth=0.8)
    ax.set_axisbelow(True)


def titulo(ax, texto: str) -> None:
    """Título alineado a la izquierda, en negrita."""
    ax.set_title(texto, loc="left", color=COLOR_TEXTO, fontweight="bold")
