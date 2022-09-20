function _(selector) {
    return document.querySelector(selector)
}

function setup() {
    let canvas = createCanvas(650, 600)
    canvas.parent("canvas-wrapper")
    background(255)
    
}

function mouseDragged() {
    
    let defaultColor = "#000";
    let type = _("#pen-pencil").checked ? "pencil":"brush"
    let size = parseInt(_("#pen-size").value)
    fill(defaultColor)
    stroke(defaultColor)
    
    ellipse(mouseX, mouseY, size, size)
    
}

function resetScreen() {
    background(255)
}

function saveDrawing(){
    //saveCanvas(canvas, "sketch", "png")
    const img = get()
    img.resize(28,28)
    console.log(img)
    loadPixels()
    
    updatePixels()

    //img.save(frameCount, '.png')
}