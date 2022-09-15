function _(selector) {
    return document.querySelector(selector)
}

function setup() {
    let canvas = createCanvas(650, 600)
    canvas.parent("canvas-wrapper")
    background(255)
}

function mouseDragged() {
    
    // only B&W images on simple MNIST 
    let defaultColor = "#000";
    let type = _("#pen-pencil").checked ? "pencil":"brush"
    let size = parseInt(_("#pen-size").value)
    fill(defaultColor)
    stroke(defaultColor)

    if(type == "pencil") {
        line(pmouseX, pmouseY, mouseX, mouseY)
    } else {
        ellipse(mouseX, mouseY, size, size)
    }
}

function resetScreen() {
    background(255)
}

function saveDrawing(){
    saveCanvas(canvas, "sketch", "png")
}