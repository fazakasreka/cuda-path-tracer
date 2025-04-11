#include "mesh.cuh"
#include <sstream>
#include <fstream>

void Box::half(Box& boxLeft, Box& boxRight, Axis axis, float boxLeftPercent) {
    boxLeft.minX = minX;
    boxLeft.maxX = maxX;
    boxLeft.minY = minY;
    boxLeft.maxY = maxY;
    boxLeft.minZ = minZ;
    boxLeft.maxZ = maxZ;
    boxRight.minX = minX;
    boxRight.maxX = maxX;
    boxRight.minY = minY;
    boxRight.maxY = maxY;
    boxRight.minZ = minZ;
    boxRight.maxZ = maxZ;
    if (axis == Axis_X) {
        boxLeft.maxX = minX + (maxX - minX) * boxLeftPercent;
        boxRight.minX = boxLeft.maxX;
    }
    if (axis == Axis_Y) {
        boxLeft.maxY = minY + (maxY - minY) * boxLeftPercent;
        boxRight.minY = boxLeft.maxY;
    }
    if (axis == Axis_Z) {
        boxLeft.maxZ = minZ + (maxZ - minZ) * boxLeftPercent;
        boxRight.minZ = boxLeft.maxZ;
    }
}

void Node::makeNodeTree(std::vector<Triangle*> triangles, int depth) {
    if (triangles.size() <= maxTriangleNum || depth > maxKdTreeHeight) {
        int size = triangles.size();
        copyToDevice(device_triangles_size, size, "device_triangles_size");
        copyToDevice(device_triangles, triangles, size, "device_triangles");
        return;
    }
    left = new Node();
    right = new Node();


    //get vertices
    int N_vertices = triangles.size() * 3;
    float* vertices = new float[N_vertices];
    for (int i = 0; i < triangles.size(); i++) {
        vertices[3 * i] = triangles[i]->a.position.axisCoordinate(axis);
        vertices[3 * i + 1] = triangles[i]->b.position.axisCoordinate(axis);
        vertices[3 * i + 2] = triangles[i]->c.position.axisCoordinate(axis);
    }

    //remove vertices out of box bounds
    int N_verces_in_bounds = 0;
    for (int i = 0; i < N_vertices; i++) {
        if (box.inBoundsOnAxis(axis, vertices[i])) N_verces_in_bounds++;
    }

    float* vertices_in_bounds = new float[N_verces_in_bounds];
    int j = 0;
    for (int i = 0; i < N_vertices; i++) {
        if (box.inBoundsOnAxis(axis, vertices[i])) {
            vertices_in_bounds[j++] = vertices[i];
        }
    }
    delete[] vertices;

    //sort verticies
    std::sort(vertices_in_bounds, vertices_in_bounds + N_verces_in_bounds, std::less<float>());


    //log2 look for optimal halfing point
    int idx_split_point = N_verces_in_bounds / 2;
    for (int i = 2; i <= triangleOptimumSearchMaxDepth + 1; i++) {
        //split
        float splitPoint = vertices_in_bounds[idx_split_point];
        box.half(left->box, right->box, axis,
            (splitPoint - box.startingPointOnAxis(axis)) / box.widthOnAxis(axis)
        );
        //count triangles in each
        int N_trinagles_left = 0, N_triangles_right = 0;
        for (auto triangle : triangles) {
            if (left->box.isTriangleInBox(triangle)) {
                N_trinagles_left++;
            }
            if (right->box.isTriangleInBox(triangle)) {
                N_triangles_right++;
            }
        }
        //decide next move
        if ((N_trinagles_left - N_triangles_right) < allowedtriangleDifference
            && -allowedtriangleDifference < (N_trinagles_left - N_triangles_right)) {
            break;
        }
        else if (N_trinagles_left > N_triangles_right) {
            idx_split_point -= N_verces_in_bounds / (2 * i);
            if (idx_split_point < 0) {
                idx_split_point = 0;
                break;
            }
        }
        else {
            idx_split_point += N_verces_in_bounds / (2 * i);
            if (idx_split_point > N_verces_in_bounds - 1) {
                idx_split_point = N_verces_in_bounds - 1;
                break;
            }
        }
    }


    //half at selected point
    box.half(left->box, right->box, axis
        , (vertices_in_bounds[idx_split_point] - box.startingPointOnAxis(axis)) / box.widthOnAxis(axis)
    );

    delete[] vertices_in_bounds;
    /////////////////////////////////////////


    //add triangles
    std::vector<Triangle*> left_triangles;
    std::vector<Triangle*> right_triangles;
    for (auto triangle : triangles) {
        if (left->box.isTriangleInBox(triangle)) {
            left_triangles.push_back(triangle);
        }
        if (right->box.isTriangleInBox(triangle)) {
            right_triangles.push_back(triangle);
        }
    }
    //change axis for next time
    left->axis = nextAxis(axis);
    right->axis = nextAxis(axis);
    left->makeNodeTree(left_triangles, depth + 1);
    right->makeNodeTree(right_triangles, depth + 1);
}

Node* Mesh::getKdTree(std::vector<Triangle*> triangles) {
    Node* kdTree = new Node();

    kdTree->box.minX = triangles[0]->a.position.x;
    kdTree->box.maxX = triangles[0]->a.position.x;
    kdTree->box.minY = triangles[0]->a.position.y;
    kdTree->box.maxY = triangles[0]->a.position.y;
    kdTree->box.minZ = triangles[0]->a.position.z;
    kdTree->box.maxZ = triangles[0]->a.position.z;
    for (auto triangle : triangles) {
        processTriangleToBox(triangle, kdTree->box);
    }

    kdTree->axis = Axis_X;
    kdTree->makeNodeTree(triangles);

    return kdTree;
}


void Mesh::processTriangleToBox(Triangle* triangle, Box& box) {
    processVertexToBox(triangle->a, box);
    processVertexToBox(triangle->b, box);
    processVertexToBox(triangle->c, box);
}
void Mesh::processVertexToBox(Vertex vertex, Box& box) {
    if (vertex.position.x < box.minX) box.minX = vertex.position.x;
    if (vertex.position.x > box.maxX) box.maxX = vertex.position.x;
    if (vertex.position.y < box.minY) box.minY = vertex.position.y;
    if (vertex.position.y > box.maxY) box.maxY = vertex.position.y;
    if (vertex.position.z < box.minZ) box.minZ = vertex.position.z;
    if (vertex.position.z > box.maxZ) box.maxZ = vertex.position.z;
}


void Mesh::uploadKdTree(Node* kdTree) {
    //save size
    int size = kdTree->getSize();
    copyToDevice(device_kdTreeSize, size, "device_kdTreeSize");

    Node *host_kdTree = new Node[size];

    std::queue<Node*> queue;
    queue.push(kdTree);
    int idx = 0;
    while (queue.size() > 0) {
        Node* currentNode = queue.front();
        queue.pop();

        while (!(currentNode->isLeaf())) {
            putNodeToIdx(currentNode, host_kdTree, idx);
            idx++;
            
            //next while
            queue.push(currentNode->left);
            currentNode = currentNode->right;
        }
        //leaf node
        putNodeToIdx(currentNode, host_kdTree, idx);
        idx++;
    }
    //upload to GPU
    copyToDevice(device_kdTree, host_kdTree, size, "device_kdTree");
}

void Mesh::putNodeToIdx(Node* currentNode, Node* array, int idx) {
    //go back to parent and update
    if (currentNode->left_child_of_parent_idx != -1) {
        array[currentNode->left_child_of_parent_idx]
            .left_index = idx;
    }
    else if (currentNode->right_child_of_parent_idx != -1) {
        array[currentNode->right_child_of_parent_idx]
            .right_index = idx;
    }

    if (!currentNode->isLeaf()) {
        //save for children where to update later
        currentNode->left->left_child_of_parent_idx = idx;
        currentNode->right->right_child_of_parent_idx = idx;
    }

    //add to array
    array[idx] = *currentNode;
}

constexpr auto POSITION = "v";
constexpr auto NORMAL = "vn";
constexpr auto FACE = "f";
constexpr auto POSITION_IDX = 0;
constexpr auto NORMAL_IDX = 2;
constexpr auto MAX_IDX = 2;
constexpr auto DEVIDER = '/';
constexpr auto EMPTY = ' ';
Mesh* readObjIntoMesh(std::string location, mat4 SRTmtx) {
	//read into these
	std::vector<vec3> positions;
	std::vector<vec3> normals;
	std::vector<int> position_idx;
	std::vector<int> normal_idx;

	//open file
	std::stringstream ss;
	std::ifstream in_file(location);
	std::string line = "";
	if (!in_file.is_open()) {
		throw "Error opening file at " + location;
	}
	//helper temps
	std::string prefix = "";
	vec3 temp;
	int i;

	//read all lines
	while (std::getline(in_file, line)) {
		//clear, read new line
		ss.clear();
		ss.str(line);
		//get prefix
		ss >> prefix;

		if (prefix == POSITION)
		{
			ss >> temp.x >> temp.y >> temp.z;
			positions.push_back(temp);
		}
		else if (prefix == NORMAL)
		{
			ss >> temp.x >> temp.y >> temp.z;
			normals.push_back(temp);
		}
		else if (prefix == FACE)
		{
			int counter = 0;
			while (ss >> i) {
				//read
				if (counter == POSITION_IDX) {
					position_idx.push_back(i - 1); //we index from 0
				}
				else if (counter == NORMAL_IDX) {
					normal_idx.push_back(i - 1); //we index from 0
				}
				//handle separators
				for (int i = 0; i < 3; i++) {
					if (ss.peek() == DEVIDER) {
						counter++;
						ss.ignore(1, DEVIDER);
					}
				}
				if (ss.peek() == EMPTY) {
					counter = 0;
					ss.ignore(1, EMPTY);
				}
			}

		}
	}

	std::vector<Triangle*> triangles;
	triangles.resize(position_idx.size() / 3);
	for (size_t i = 0; i < triangles.size(); i++) {
		int vertexAIdx = i * 3;
		int vertexBIdx = i * 3 + 1;
		int vertexCIdx = i * 3 + 2;
		vec3 posA = (vec4(positions[position_idx[vertexAIdx]], 1) * SRTmtx ).xyz();
		vec3 normA = (vec4(normals[normal_idx[vertexAIdx]], 0) * SRTmtx.inverse().transpose()).xyz().normalize();
		vec3 posB = (vec4(positions[position_idx[vertexBIdx]], 1) * SRTmtx).xyz();
		vec3 normB = (vec4(normals[normal_idx[vertexBIdx]], 0) * SRTmtx.inverse().transpose()).xyz().normalize();
		vec3 posC = (vec4(positions[position_idx[vertexCIdx]], 1) * SRTmtx).xyz();
		vec3 normC = (vec4(normals[normal_idx[vertexCIdx]], 0) * SRTmtx.inverse().transpose()).xyz().normalize();

		triangles[i] = new Triangle(
			Vertex(posA, normA),
			Vertex(posB, normB),
			Vertex(posC, normC),
			nullptr
		);
	}
	return new Mesh(triangles);
}