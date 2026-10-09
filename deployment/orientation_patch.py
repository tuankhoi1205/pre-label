"""Backport a display-only cuboid orientation switch to the pinned CVAT UI."""

ORIENTATION_PATCHES = {
    "cvat-ui/src/actions/settings-actions.ts": [
        ("    CHANGE_SHAPES_SHOW_PROJECTIONS = 'CHANGE_SHAPES_SHOW_PROJECTIONS',",
         "    CHANGE_SHAPES_SHOW_PROJECTIONS = 'CHANGE_SHAPES_SHOW_PROJECTIONS',\n"
         "    CHANGE_SHAPES_SHOW_ORIENTATION = 'CHANGE_SHAPES_SHOW_ORIENTATION',"),
        ("export function changeShowProjections(showProjections: boolean): AnyAction {",
         """// PRELABEL_3D_ORIENTATION: appearance only; never edits annotation geometry.
export function changeShowOrientation(showOrientation: boolean): AnyAction {
    return {
        type: SettingsActionTypes.CHANGE_SHAPES_SHOW_ORIENTATION,
        payload: { showOrientation },
    };
}

export function changeShowProjections(showProjections: boolean): AnyAction {"""),
    ],
    "cvat-ui/src/reducers/settings-reducer.ts": [
        ("        showProjections: false,", "        showProjections: false,\n        showOrientation: false,"),
        ("        case SettingsActionTypes.CHANGE_SHAPES_SHOW_PROJECTIONS: {", """        case SettingsActionTypes.CHANGE_SHAPES_SHOW_ORIENTATION: {
            return {
                ...state,
                shapes: { ...state.shapes, showOrientation: action.payload.showOrientation },
            };
        }
        case SettingsActionTypes.CHANGE_SHAPES_SHOW_PROJECTIONS: {"""),
    ],
    "cvat-ui/src/reducers/index.ts": [
        ("    showProjections: boolean;", "    showProjections: boolean;\n    showOrientation: boolean;"),
    ],
    "cvat-ui/src/components/annotation-page/appearance-block.tsx": [
        ("    changeShowProjections as changeShowProjectionsAction,",
         "    changeShowProjections as changeShowProjectionsAction,\n"
         "    changeShowOrientation as changeShowOrientationAction,"),
        ("    showProjections: boolean;", "    showProjections: boolean;\n    showOrientation: boolean;"),
        ("    changeShowProjections(event: CheckboxChangeEvent): void;",
         "    changeShowProjections(event: CheckboxChangeEvent): void;\n"
         "    changeShowOrientation(event: CheckboxChangeEvent): void;"),
        ("                colorBy, opacity, selectedOpacity, outlined, outlineColor, showBitmap, showProjections,",
         "                colorBy, opacity, selectedOpacity, outlined, outlineColor, showBitmap,\n"
         "                showProjections, showOrientation,"),
        ("        showProjections,\n        workspace,", "        showProjections,\n        showOrientation,\n        workspace,"),
        ("        changeShowProjections(event: CheckboxChangeEvent): void {", """        changeShowOrientation(event: CheckboxChangeEvent): void {
            dispatch(changeShowOrientationAction(event.target.checked));
        },
        changeShowProjections(event: CheckboxChangeEvent): void {"""),
        ("        showProjections,\n        collapseAppearance,", "        showProjections,\n        showOrientation,\n        collapseAppearance,"),
        ("        changeShowProjections,\n        jobInstance,", "        changeShowProjections,\n        changeShowOrientation,\n        jobInstance,"),
        ("                        {is2D && (\n                            <Checkbox\n                                className='cvat-appearance-bitmap-checkbox'", """                        {!is2D && (
                            <Checkbox
                                className='cvat-appearance-cuboid-orientation-checkbox'
                                onChange={changeShowOrientation}
                                checked={showOrientation}
                                title='Local axes: X red (heading), Y green, Z blue'
                            >
                                Cuboid orientation
                            </Checkbox>
                        )}
                        {is2D && (
                            <Checkbox
                                className='cvat-appearance-bitmap-checkbox'"""),
    ],
    "cvat-ui/src/components/annotation-page/canvas/views/canvas3d/canvas-wrapper3D.tsx": [
        ("    colorBy: ColorBy;", "    colorBy: ColorBy;\n    showOrientation: boolean;"),
        ("                opacity, colorBy, selectedOpacity, outlined, outlineColor,",
         "                opacity, colorBy, selectedOpacity, outlined, outlineColor, showOrientation,"),
        ("        outlineColor,\n        activeLabelID,", "        outlineColor,\n        showOrientation,\n        activeLabelID,"),
        ("            selectedOpacity,\n            colorBy,\n        });",
         "            selectedOpacity,\n            colorBy,\n            showOrientation,\n        });"),
        ("    }, [opacity, outlined, outlineColor, selectedOpacity, colorBy]);",
         "    }, [opacity, outlined, outlineColor, selectedOpacity, colorBy, showOrientation]);"),
        ("        colorBy,\n        contextMenuVisibility,", "        colorBy,\n        showOrientation,\n        contextMenuVisibility,"),
    ],
    "cvat-canvas3d/src/typescript/canvas3dModel.ts": [
        ("    colorBy: string;", "    colorBy: string;\n    showOrientation?: boolean;"),
        ("                outlineColor: '#000000',", "                outlineColor: '#000000',\n                showOrientation: false,"),
        ("        this.notify(UpdateReasons.SHAPES_CONFIG_UPDATED);", """        if (typeof shapeProperties.showOrientation === 'boolean') {
            this.data.shapeProperties.showOrientation = shapeProperties.showOrientation;
        }

        this.notify(UpdateReasons.SHAPES_CONFIG_UPDATED);"""),
    ],
    "cvat-canvas3d/src/typescript/cuboid.ts": [
        ("    public setPosition(x: number, y: number, z: number): void {", """    // PRELABEL_3D_ORIENTATION: local +X follows the stored cuboid yaw/rotation.
    // Child helpers inherit all transforms but cannot intercept selection rays.
    public setOrientationVisible(visible: boolean): void {
        [ViewType.PERSPECTIVE, ViewType.TOP, ViewType.SIDE, ViewType.FRONT].forEach((view): void => {
            const mesh = (this as Indexable)[view] as THREE.Mesh;
            let axes = mesh.getObjectByName('prelabelCuboidOrientation');
            if (!axes && visible) {
                axes = new THREE.Group();
                axes.name = 'prelabelCuboidOrientation';
                const directions = [
                    new THREE.Vector3(1, 0, 0),
                    new THREE.Vector3(0, 1, 0),
                    new THREE.Vector3(0, 0, 1),
                ];
                const colors = [0xff3333, 0x33dd55, 0x3399ff];
                directions.forEach((direction, index): void => {
                    const arrow = new THREE.ArrowHelper(direction, new THREE.Vector3(), 0.75, colors[index], 0.18, 0.12);
                    arrow.traverse((child): void => {
                        child.raycast = (): void => {};
                    });
                    axes.add(arrow);
                });
                mesh.add(axes);
            }
            if (axes) axes.visible = visible;
        });
    }

    public setPosition(x: number, y: number, z: number): void {"""),
    ],
    "cvat-canvas3d/src/typescript/canvas3dView.ts": [
        ("        cuboid.attachCameraReference();",
         "        cuboid.attachCameraReference();\n"
         "        cuboid.setOrientationVisible(Boolean(this.model.data.shapeProperties.showOrientation));"),
        ("                if (config.outlined) {",
         "                cuboid.setOrientationVisible(Boolean(config.showOrientation));\n\n"
         "                if (config.outlined) {"),
    ],
}
